"""Personal memory retriever agent for retrieving personal memories through vector search."""

from ..base_memory_agent import BaseMemoryAgent
from ....core.enumeration import Role, MemoryType
from ....core.op import BaseTool
from ....core.schema import Message
from ....core.utils import format_messages


class PersonalRetriever(BaseMemoryAgent):
    """Retrieve personal memories through vector search and history reading.

    clear && python benchmark/halumem/eval_reme.py \
        --data_path /Users/yuli/workspace/HaluMem/data/HaluMem-Medium.jsonl \
        --reme_model_name qwen3.5-plus \
        --batch_size 10000 \
        --algo_version default

    📊 Question Answering (with LLM answer):
      Correct (all):       0.8537
      Hallucination (all): 0.1159
      Omission (all):      0.0305
      Correct (valid):     0.8537
      Hallucination (valid): 0.1159
      Omission (valid):    0.0305
      Valid/Total:         164/164

    📊 Question Answering (with original memories):
      Correct (all):       0.9085
      Hallucination (all): 0.0671
      Omission (all):      0.0244
      Correct (valid):     0.9085
      Hallucination (valid): 0.0671
      Omission (valid):    0.0244
      Valid/Total:         164/164
    """

    memory_type: MemoryType = MemoryType.PERSONAL

    def __init__(self, return_memory_nodes: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.return_memory_nodes: bool = return_memory_nodes

    async def build_messages(self) -> list[Message]:
        if self.context.get("query"):
            context = self.context.query
        elif self.context.get("messages"):
            context = self.description + "\n" + format_messages(self.context.messages)
        else:
            raise ValueError("input must have either `query` or `messages`")

        read_all_profiles_tool: BaseTool | None = self.pop_tool("read_all_profiles")
        if read_all_profiles_tool is not None:
            all_profiles = await read_all_profiles_tool.call(
                memory_target=self.memory_target,
                service_context=self.service_context,
            )
        else:
            all_profiles = ""

        return [
            Message(
                role=Role.USER,
                content=self.prompt_format(
                    prompt_name="user_message",
                    memory_type=self.memory_type.value,
                    memory_target=self.memory_target,
                    user_profile=all_profiles,
                    context=context.strip(),
                ),
            ),
        ]

    async def _acting_step(
        self,
        assistant_message: Message,
        tools: list[BaseTool],
        step: int,
        stage: str = "",
        **kwargs,
    ) -> tuple[list[BaseTool], list[Message]]:
        """Execute tool calls with memory context."""
        return await super()._acting_step(
            assistant_message,
            tools,
            step,
            memory_type=self.memory_type.value,
            memory_target=self.memory_target,
            retrieved_nodes=self.retrieved_nodes,
            **kwargs,
        )

    async def execute(self):
        result = await super().execute()
        if self.return_memory_nodes:
            result["answer"] = "\n".join(
                [
                    n.format(
                        include_memory_id=False,
                        include_when_to_use=False,
                        include_content=True,
                        include_message_time=True,
                        ref_memory_id_key="",
                    )
                    for n in self.retrieved_nodes
                ],
            )

        result["retrieved_nodes"] = self.retrieved_nodes
        return result
