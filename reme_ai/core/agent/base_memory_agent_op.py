import asyncio
from abc import ABC
from typing import List, Dict, Optional, Any

from loguru import logger

from .. import BaseAsyncToolOp
from ..enumeration import Role, MemoryType
from ..schema import Message, ToolCall
from ..tool import ThinkToolOp


class BaseMemoryAgentOp(BaseAsyncToolOp, ABC):
    memory_type: MemoryType | None = None

    def __init__(
            self,
            max_steps: int = 20,
            tool_call_interval: float = 0,
            add_think_tool: bool = False,  # only for instruct model
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_steps: int = max_steps
        self.tool_call_interval: float = tool_call_interval
        self.add_think_tool: bool = add_think_tool

    def build_tool_call(self) -> ToolCall:
        raise NotImplementedError("Subclasses must implement `build_tool_call`")

    async def build_tool_op_dict(self) -> dict:
        tool_op_dict: Dict[str, BaseAsyncToolOp] = {
            op.tool_call.name: op for op in self.ops.values() if isinstance(op, BaseAsyncToolOp)
        }
        for op in tool_op_dict.values():
            op.language = self.language
        return tool_op_dict

    async def build_messages(self) -> List[Message]:
        raise NotImplementedError("Subclasses must implement `build_messages`")

    @staticmethod
    def format_messages(messages: List[Dict[str, Any]]):
        messages = [Message(**x) for x in messages if isinstance(x, dict)]
        messages = [x for x in messages if x.role is not Role.SYSTEM]
        messages_context = "\n".join([x.format_message(
            add_time_created=True,
            use_name_first=True,
            add_reasoning_content=True,
            add_tool_calls=True,
        ) for x in messages])
        return messages_context

    async def _reasoning_step(
            self,
            messages: List[Message],
            tool_op_dict: Dict[str, BaseAsyncToolOp],
            step: int,
    ) -> tuple[Message, bool]:
        assistant_message: Message = await self.llm.achat(
            messages=messages,
            tools=[op.tool_call for op in tool_op_dict.values()],
        )
        messages.append(assistant_message)
        logger.info(f"step{step + 1}.assistant={assistant_message.model_dump_json()}")
        should_act = bool(assistant_message.tool_calls)
        return assistant_message, should_act

    async def _acting_step(
            self,
            assistant_message: Message,
            tool_op_dict: Dict[str, BaseAsyncToolOp],
            think_op: Optional[BaseAsyncToolOp],
            step: int,
    ) -> List[Message]:
        if not assistant_message.tool_calls:
            return []

        op_list: List[BaseAsyncToolOp] = []
        has_think_tool_flag: bool = False
        tool_result_messages: List[Message] = []

        for j, tool_call in enumerate(assistant_message.tool_calls):
            if think_op is not None and tool_call.name == think_op.tool_call.name:
                has_think_tool_flag = True

            if tool_call.name not in tool_op_dict:
                logger.exception(f"unknown tool_call.name={tool_call.name}")
                continue

            logger.info(f"step{step + 1}.{j} submit tool_calls={tool_call.name} argument={tool_call.argument_dict}")
            op_copy: BaseAsyncToolOp = tool_op_dict[tool_call.name].copy()
            op_copy.tool_call.id = tool_call.id
            op_list.append(op_copy)
            self.submit_async_task(op_copy.async_call,
                                   memory_type=self.memory_type,
                                   memory_target=self.memory_target,
                                   ref_memory_id=self.ref_memory_id,
                                   workspace_id=self.workspace_id,
                                   author=self.author,
                                   **tool_call.argument_dict)
            if self.tool_call_interval > 0:
                await asyncio.sleep(self.tool_call_interval)

        await self.join_async_task()

        for j, op in enumerate(op_list):
            tool_result = str(op.output)
            tool_message = Message(
                role=Role.TOOL,
                content=tool_result,
                tool_call_id=op.tool_call.id,
            )
            tool_result_messages.append(tool_message)
            logger.info(f"step{step + 1}.{j} join tool_result={tool_result[:200]}...\n\n")

        if self.add_think_tool:
            if not has_think_tool_flag:
                tool_op_dict["think_tool"] = think_op
            else:
                tool_op_dict.pop("think_tool", None)

        return tool_result_messages

    async def async_execute(self):
        tool_op_dict = await self.build_tool_op_dict()
        if self.add_think_tool:
            think_op = ThinkToolOp(language=self.language)
            tool_op_dict["think_tool"] = think_op
        else:
            think_op = None

        messages = await self.build_messages()
        for i, message in enumerate(messages):
            logger.info(f"step0.{i} {message.role} {message.name or ''} {message.simple_dump()}")

        for step in range(self.max_steps):
            assistant_message, should_act = await self._reasoning_step(messages, tool_op_dict, step)

            if not should_act:
                break

            tool_result_messages = await self._acting_step(assistant_message, tool_op_dict, think_op, step)
            messages.extend(tool_result_messages)

        if messages:
            self.set_output(messages[-1].content)
        else:
            self.set_output("")
        self.context.response.metadata["messages"] = messages

    @property
    def memory_target(self) -> str:
        return self.input_dict.get("memory_target", "")

    @property
    def ref_memory_id(self) -> str:
        return self.input_dict.get("ref_memory_id", "")

    @property
    def workspace_id(self) -> str:
        return self.input_dict.get("workspace_id", "default")

    @property
    def author(self) -> str:
        return self.llm_config.model_name
