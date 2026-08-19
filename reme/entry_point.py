"""Safe loading boundary for third-party Python entry points."""

from importlib import metadata
from typing import Any


def load_entry_point(entry: metadata.EntryPoint, *, invoke: bool = False) -> Any:
    """Load and optionally invoke an entry point without retaining registrations."""
    # Import lazily so config parser imports do not pull in the component graph.
    from .components.component_registry import R

    with R.preserve(allow_mutation=True):
        loaded = entry.load()
        return loaded() if invoke and callable(loaded) else loaded
