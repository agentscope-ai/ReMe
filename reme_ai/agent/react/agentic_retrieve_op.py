"""
Async React agent operator tailored for retrieval workflows.

This module implements a ReAct (Reasoning + Acting) agent pattern that combines
language model reasoning with tool execution. The agent iteratively:
1. Processes and manages conversation context (compaction/compression)
2. Generates reasoning and tool calls via LLM
3. Executes tools (e.g., file search, reading)
4. Incorporates tool results back into the conversation
5. Repeats until a final answer is reached or max_steps is exceeded

The agent is specifically designed for RAG (Retrieval-Augmented Generation) workflows,
providing context management capabilities to handle long conversations efficiently.
"""

from typing import Dict, List

from flowllm.core.context import C
from flowllm.core.enumeration import Role
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import ToolCall, Message
from loguru import logger


@C.register_op()
class AgenticRetrieveOp(BaseAsyncToolOp):
    """
    ReAct agent that exposes RAG-friendly tools and context policies.

    This agent implements the ReAct pattern where the LLM alternates between:
    - Reasoning: Analyzing the problem and deciding what to do
    - Acting: Executing tools to gather information
    - Observing: Processing tool results and updating understanding

    The agent supports sophisticated context management to handle long conversations
    through compaction (storing large tool messages externally) and compression
    (LLM-based summarization of message history).

    Available tools:
    - GrepOp: Search for patterns in files
    - ReadFileOp: Read file contents

    The agent automatically manages context size by offloading large messages and
    compressing conversation history when token limits are approached.
    """

    file_path: str = __file__

    def __init__(
        self,
        llm: str = "qwen3_30b_instruct",
        max_steps: int = 5,
        **kwargs,
    ):
        """
        Initialize the agent runtime configuration.

        Args:
            llm: Identifier for the language model to use. Defaults to "qwen3_30b_instruct".
            max_steps: Maximum number of reasoning-action cycles before stopping.
                      Each cycle includes: context management -> LLM reasoning -> tool execution.
                      Defaults to 5 steps.
            **kwargs: Additional arguments passed to the base BaseAsyncToolOp class.
        """
        super().__init__(llm=llm, **kwargs)
        # Maximum number of ReAct iterations (reasoning + tool execution cycles)
        self.max_steps: int = max_steps

    def build_tool_call(self) -> ToolCall:
        """
        Expose metadata describing how to invoke the agent.

        This method defines the tool schema that other components use to invoke
        this agent. It specifies all input parameters including conversation messages
        and context management configuration.

        Returns:
            ToolCall: A schema object describing the agent's interface, including
                     all parameters for message handling and context management.
        """
        return ToolCall(
            **{
                "description": "A React agent that answers user queries.",
                "input_schema": {
                    "messages": {
                        "type": "array",
                        "description": "messages",
                        "required": True,
                    },
                    "context_manage_mode": {
                        "type": "string",
                        "description": "Context management mode: 'compact' (only compacts tool messages), 'compress' "
                        "(only LLM-based compression), 'auto' (compaction first then compression if "
                        "needed). Defaults to 'auto'.",
                        "required": True,
                        "enum": ["compact", "compress", "auto"],
                    },
                    "max_total_tokens": {
                        "type": "integer",
                        "description": "Maximum token threshold for triggering compression/compaction. For compaction "
                        "this is total tokens; for compression this excludes keep_recent_count and "
                        "system messages. Defaults to 20000.",
                        "required": False,
                    },
                    "max_tool_message_tokens": {
                        "type": "integer",
                        "description": "Maximum token count per tool message before compaction applies. Exceeding "
                        "messages store full content externally with a preview in context. Defaults "
                        "to 2000.",
                        "required": False,
                    },
                    "group_token_threshold": {
                        "type": "integer",
                        "description": "Maximum tokens per compression group for LLM-based compression. None/0 "
                        "compresses all messages together. Oversized messages form their own group. "
                        "Used in 'compress' or 'auto' mode.",
                        "required": False,
                    },
                    "keep_recent_count": {
                        "type": "integer",
                        "description": "Number of recent messages preserved without compression/compaction. Defaults "
                        "to 1 for compaction and 2 for compression.",
                        "required": False,
                    },
                    "store_dir": {
                        "type": "string",
                        "description": "Directory for storing offloaded contents. Required for compaction/compression "
                        "to save full tool messages and compressed groups.",
                        "required": False,
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Chat session identifier for naming stored files. Defaults to auto-generated "
                        "UUID if omitted.",
                        "required": False,
                    },
                },
            },
        )

    async def async_execute(self):
        """
        Main execution loop implementing the ReAct (Reasoning + Acting) pattern.

        This method orchestrates the iterative agent workflow:
        1. Context Management: Compact/compress conversation history if needed
        2. Reasoning: LLM analyzes context and decides on actions (may include tool calls)
        3. Acting: Execute requested tools in parallel
        4. Observing: Incorporate tool results back into conversation
        5. Repeat until final answer (no tool calls) or max_steps reached

        The loop continues until:
        - The LLM produces a response without tool calls (final answer)
        - Maximum number of steps (max_steps) is reached

        Each iteration is called a "round" and represents one reasoning-action cycle.
        """
        # Import tool operators that the agent can use
        from reme_ai.context.file_tool import GrepOp, ReadFileOp
        from reme_ai.context.offload import ContextOffloadOp
        from reme_ai.context.file_tool import BatchWriteFileOp

        # Initialize available tools for the agent
        # GrepOp: Search for patterns/text in files (useful for code search)
        grep_op = GrepOp(language=self.language)
        # ReadFileOp: Read and return file contents
        read_file_op = ReadFileOp(language=self.language)

        # Create a dictionary mapping tool names to their operator instances
        # This allows quick lookup when the LLM requests a specific tool
        tool_op_dict: Dict[str, BaseAsyncToolOp] = {
            grep_op.tool_call.name: grep_op,
            read_file_op.tool_call.name: read_file_op,
        }

        # Convert input messages to Message objects for processing
        messages = [Message(**x) for x in self.context.messages]

        # Extract context management parameters from input, excluding messages
        # These will be passed to the context management pipeline
        context_kwargs = self.input_dict.copy()
        context_kwargs.pop("messages", None)

        # Main ReAct loop: iterate up to max_steps times
        for i in range(self.max_steps):
            # Step 1: Context Management Phase
            # Create a pipeline: ContextOffloadOp (compacts/compresses) -> BatchWriteFileOp (saves offloaded content)
            # The >> operator chains these operations together
            op = ContextOffloadOp() >> BatchWriteFileOp()

            # Apply context management to current message history
            # This may compact large tool messages or compress old messages based on context_manage_mode
            await op.async_call(messages=[x.simple_dump() for x in messages], **context_kwargs)

            # Update messages with the processed/optimized version from context management
            # Large messages may now reference external files instead of containing full content
            messages = [Message(**x) for x in op.context.response.answer]

            # Step 2: Reasoning Phase
            # LLM analyzes the conversation context and decides what to do
            # It may generate a final answer or request tool calls to gather more information
            assistant_message: Message = await self.llm.achat(
                messages=messages,  # Current conversation history (possibly optimized)
                tools=[op.tool_call for op in tool_op_dict.values()],  # Available tools the LLM can use
            )

            # Add the LLM's response to the conversation history
            messages.append(assistant_message)
            logger.info(f"round{i + 1}.assistant={assistant_message.model_dump_json()}")

            # Step 3: Check if we have a final answer
            # If the LLM didn't request any tools, it has provided a final answer
            # Exit the loop as the agent's work is complete
            if not assistant_message.tool_calls:
                break

            # Step 4: Acting Phase - Execute requested tools
            # The LLM requested one or more tools to gather information
            # We'll execute them in parallel for efficiency
            op_list: List[BaseAsyncToolOp] = []

            # Process each tool call requested by the LLM
            for j, tool_call in enumerate(assistant_message.tool_calls):
                # Validate that the requested tool exists in our available tools
                if tool_call.name not in tool_op_dict:
                    logger.exception(f"unknown tool_call.name={tool_call.name}")
                    continue

                logger.info(f"round{i + 1}.{j} submit tool_calls={tool_call.name} argument={tool_call.argument_dict}")

                # Create a copy of the tool operator for this specific tool call
                # Each tool call needs its own operator instance with the correct call ID
                op_copy: BaseAsyncToolOp = tool_op_dict[tool_call.name].copy()
                op_copy.tool_call.id = tool_call.id  # Match the ID from LLM's tool call request
                op_list.append(op_copy)

                # Submit the tool execution as an async task (runs in parallel with other tools)
                # The op_copy instance is used to execute the tool with the specific call ID and arguments
                self.submit_async_task(op_copy.async_call, **tool_call.argument_dict)

            # Wait for all submitted tool executions to complete
            # This ensures we have all results before proceeding
            await self.join_async_task()

            # Step 5: Observing Phase - Incorporate tool results into conversation
            # Process each completed tool execution and add results to message history
            for j, op in enumerate(op_list):
                # Extract the tool execution result as a string
                tool_result = str(op.output)

                # Create a tool message with the result, linked to the original tool call via ID
                # This allows the LLM to associate results with the specific tool calls it made
                tool_message = Message(role=Role.TOOL, content=tool_result, tool_call_id=op.tool_call.id)
                messages.append(tool_message)

                # Log the tool result (truncated to first 200 chars for readability)
                logger.info(f"round{i + 1}.{j} join tool_result={tool_result[:200]}...\n\n")

            # Loop continues: next iteration will process these tool results,
            # manage context again if needed, and let LLM reason about the new information

        # After loop completes, set the final output
        # The last message should be the LLM's final answer (without tool calls)
        self.set_output(messages[-1].content)

        # Store the complete conversation history in the context response
        # This includes all reasoning steps, tool calls, and tool results
        self.context.response.answer = messages
