"""Portable provenance retained with committed proactive inputs and notifications."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceRef(BaseModel):
    """Freeze source display information without changing the original Event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    name: str
    trigger_type: Literal["event", "time"]


class InputTrace(BaseModel):
    """An execution attempt is independent of native sessions and turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    event_ids: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


def merge_traces(traces):
    """Merge batch provenance without losing any source, event or execution identity."""
    refs, events, runs = {}, {}, {}
    for trace in traces:
        if trace is not None:
            refs.update((ref.source_id, ref.model_dump(mode="json")) for ref in trace.source_refs)
            events.update(dict.fromkeys(trace.event_ids))
            runs[trace.run_id] = None
    return {"source_refs": list(refs.values()), "event_ids": list(events), "run_ids": list(runs)}
