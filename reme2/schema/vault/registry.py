"""Map vault category → strict pydantic model for validation.

Used by the Maintainer to derive lint rules from schema instead of
hardcoding. Two business object schemas:
    - event              → Event
    - everything else    → Topic (judgment subset enforced via model_validator)
"""

from typing import get_args

from pydantic import BaseModel

from .event import Event
from .topic import Topic, TopicCategory

_REGISTRY: dict[str, type[BaseModel]] = {"event": Event}
for _cat in get_args(TopicCategory):
    _REGISTRY[_cat] = Topic

ALL_KNOWN_CATEGORIES: frozenset[str] = frozenset(_REGISTRY.keys())


def schema_for(category: str | None) -> type[BaseModel] | None:
    """Return the pydantic schema bound to a category, or None for unknown."""
    if not category:
        return None
    return _REGISTRY.get(category)
