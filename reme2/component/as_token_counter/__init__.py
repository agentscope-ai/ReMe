"""AgentScope TokenCounter wrappers."""

from agentscope.token import TokenCounterBase

from .estimate_token_counter import EstimatedTokenCounter
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsTokenCounter(BaseComponent):
    """Base wrapper for AgentScope token counters.

    Subclasses should implement _start() to initialize self.token_counter.
    """

    component_type = ComponentEnum.AS_TOKEN_COUNTER

    def __init__(self, **kwargs) -> None:
        """Initialize with token counter configuration kwargs."""
        super().__init__(**kwargs)
        self.token_counter: TokenCounterBase | None = None

    async def _start(self, app_context=None) -> None:
        """Initialize the token counter. Override in subclasses."""

    async def _close(self) -> None:
        """Release token counter resources."""
        self.token_counter = None


@R.register("estimated")
class EstimatedAsTokenCounter(BaseAsTokenCounter):
    """Estimated token counter using character-based estimation."""

    async def _start(self, app_context=None) -> None:
        """Initialize the estimated token counter."""
        self.token_counter = EstimatedTokenCounter(**self.kwargs)


__all__ = [
    "BaseAsTokenCounter",
    "EstimatedAsTokenCounter",
]