"""ResourceEntry — one row in ``resource/<date>/meta.json``.

Each entry describes a single passively-received asset that the
``upload`` step has copied into the day's bucket. The bucket is a
flat folder named after the receive date; the entry preserves the
provenance (channel + source) so the main agent can later cite the
asset with full context.

Reme core reserves nothing on this schema — it's a service-tier
convention owned by the upload + synchronizer flow. The Pydantic
model is here only so the file format stays stable for
``assemble_day_md`` and the upcoming consumers in synchronizer.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResourceEntry(BaseModel):
    """One asset under ``resource/<date>/``."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Final file name inside resource/<date>/ (after collision-dedupe).")
    channel: str = Field(description="Inbound channel identifier (wechat / email / browser / api / ...).")
    source: str = Field(default="", description="Free-form origin within the channel (group name, sender, URL, ...).")
    received_at: str = Field(default="", description="ISO 8601 timestamp when the asset was received.")
    description: str = Field(default="", description="One-line human-readable description.")
