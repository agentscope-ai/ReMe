"""AgentScope token counter wrappers."""

from agentscope.token import TokenCounterBase

from .estimate_token_counter import EstimatedTokenCounter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsTokenCounter(BaseComponent):
    """Base wrapper for AgentScope token counters. Builds ``self.token_counter`` in ``_start``."""

    component_type = ComponentEnum.AS_TOKEN_COUNTER

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.token_counter: TokenCounterBase | None = None

    async def _close(self) -> None:
        self.token_counter = None


@R.register("estimated")
class EstimatedAsTokenCounter(BaseAsTokenCounter):
    """Character-based estimated token counter — fast but approximate."""

    async def _start(self) -> None:
        self.token_counter = EstimatedTokenCounter(**self.kwargs)


__all__ = [
    "BaseAsTokenCounter",
    "EstimatedAsTokenCounter",
]
