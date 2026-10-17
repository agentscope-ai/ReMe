"""Provider-independent, lifecycle-managed probabilistic decision models."""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Protocol

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class DecisionResponse(Protocol):
    """JSON-serializable dataclass response shared by decision model adapters.

    Answers are keyed by question name. Binary answers expose
    ``type="binary_answer"`` and a ``probability`` between zero and one.
    Usage may be a JSON object, a dataclass, or None. Providers may add
    answer types, usage fields, and response metadata.
    """

    model: str
    content: Mapping[str, Any]
    usage: Any | None


class BaseAsDecisionModel(BaseComponent):
    """Evaluate shared text or JSON state with named probabilistic questions."""

    component_type = ComponentEnum.AS_DECISION_MODEL

    @abstractmethod
    async def __call__(
        self,
        state: str | dict,
        questions: Mapping[str, Any],
    ) -> DecisionResponse:
        """Return typed answers; concrete components own provider resources."""
