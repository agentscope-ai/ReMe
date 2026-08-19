"""Safe loading boundary for third-party Python entry points."""

from importlib import metadata
from typing import Any


def find_entry_point(group: str, name: str, *, provider: str) -> metadata.EntryPoint | None:
    """Return the sole matching entry point, rejecting ambiguous providers."""
    entries = list(metadata.entry_points().select(group=group, name=name))
    if len(entries) > 1:
        values = ", ".join(sorted(entry.value for entry in entries))
        raise ValueError(f"{provider} '{name}' has multiple installed providers: {values}")
    return entries[0] if entries else None


def load_entry_point(entry: metadata.EntryPoint, *, invoke: bool = False) -> Any:
    """Load and optionally invoke an entry point without retaining registrations."""
    # Import lazily so config parser imports do not pull in the component graph.
    from .components.component_registry import R

    with R.preserve(allow_mutation=True):
        loaded = entry.load()
        return loaded() if invoke and callable(loaded) else loaded
