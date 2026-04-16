"""ReMe CLI application entry point."""

import asyncio
import sys
from pathlib import Path

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.token import HuggingFaceTokenCounter, TokenCounterBase
from agentscope.tool import Toolkit

from .application import Application
from .component import R
from .component.runtime_context import RuntimeContext
from .config import parse_args
from .enumeration import ComponentEnum
from .file_based.summarizer import Summarizer


class ReMe(Application):
    """ReMe memory management application."""

    async def summary_memory(
            self,
            messages: list[Msg],
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase | HuggingFaceTokenCounter = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            max_input_length: float = 128 * 1024,
            compact_ratio: float = 0.7,
            timezone: str | None = None,
            add_thinking_block: bool = True,
    ) -> str:
        """Summarize and compact memory messages.

        Args:
            messages: List of AgentScope messages to summarize.
            as_llm: LLM model name or instance.
            as_llm_formatter: Formatter name or instance.
            as_token_counter: Token counter name or instance.
            toolkit: Optional toolkit for the summarizer agent.
            language: Language for prompts (zh or en).
            max_input_length: Maximum input token length.
            compact_ratio: Ratio of max_input_length to use as compact threshold.
            timezone: Optional timezone for date formatting.
            add_thinking_block: Whether to include thinking blocks.

        Returns:
            Summarized memory string.
        """
        working_dir = Path(self.config.working_dir).absolute()
        memory_dir = working_dir / "memory"
        memory_compact_threshold = int(max_input_length * compact_ratio)

        # Resolve token counter - use provided instance or create default
        token_counter_instance = None
        if isinstance(as_token_counter, HuggingFaceTokenCounter):
            token_counter_instance = as_token_counter
        else:
            token_counter_instance = HuggingFaceTokenCounter()

        summarizer = Summarizer(
            working_dir=str(working_dir),
            memory_dir=str(memory_dir),
            memory_compact_threshold=memory_compact_threshold,
            toolkit=toolkit,
            timezone=timezone,
            add_thinking_block=add_thinking_block,
            as_token_counter=token_counter_instance,
            language=language,
            as_llm=as_llm if isinstance(as_llm, str) else "default",
            as_llm_formatter=as_llm_formatter if isinstance(as_llm_formatter, str) else "default",
        )

        context = RuntimeContext(
            messages=messages,
            application_context=self.context,
        )

        result = await summarizer(context=context)
        return result or ""

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> str:
        """Search memory for relevant entries."""
        from .file_based.memory_search import MemorySearch
        try:
            search_step = MemorySearch()
            self.logger.info(f"Running memory search with {query} {max_results} {min_score}")
            return await search_step(query=query, max_results=max_results, min_score=min_score)
        except Exception as e:
            return str(e)

    async def dream(
            self,
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            timezone: str | None = None,
    ) -> str:
        """Process and consolidate memories in background."""

    async def proactive(
            self,
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            timezone: str | None = None,
    ) -> str:
        """Generate proactive memory insights."""


def main():
    """Entry point for ReMe CLI."""
    action, config = parse_args(sys.argv[1:])
    if action == "start":
        reme = ReMe(**config)
        reme.run_app()

    else:
        backend: str = config.pop("backend", "http")
        client_cls = R.get(ComponentEnum.CLIENT, backend)
        client = client_cls(action=action, **config)
        asyncio.run(client())


if __name__ == "__main__":
    main()
