import asyncio
import sys

from agentscope.formatter import FormatterBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.token import TokenCounterBase
from agentscope.tool import Toolkit, ToolResponse

from .application import Application
from .component import R
from .config import parse_args
from .enumeration import ComponentEnum


class ReMe(Application):

    async def summary_memory(
            self,
            messages: list[Msg],
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            max_input_length: float = 128 * 1024,
            compact_ratio: float = 0.7,
            timezone: str | None = None,
            add_thinking_block: bool = True,
    ) -> str:
        ...

    async def memory_search(self, query: str, max_results: int = 5, min_score: float = 0.1) -> ToolResponse:
        ...

    async def dream(
            self,
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            timezone: str | None = None,
    ) -> str:
        ...

    async def proactive(
            self,
            as_llm: str | ChatModelBase = "default",
            as_llm_formatter: str | FormatterBase = "default",
            as_token_counter: str | TokenCounterBase = "default",
            toolkit: Toolkit | None = None,
            language: str = "zh",
            timezone: str | None = None,
    ) -> str:
        ...


def main():
    action, config = parse_args(sys.argv[1:])
    if action == "app":
        reme = ReMe(**config)
        reme.run_app()

    else:
        backend: str = config.pop("backend", "http")
        client_cls = R.get(ComponentEnum.CLIENT, backend)
        client = client_cls(action=action, **config)
        asyncio.run(client())


if __name__ == "__main__":
    main()
