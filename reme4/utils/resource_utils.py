"""Resource bucket helpers — pure functions for the day's meta.json
and its derived ``<date>.md`` view.

``resource/<date>/`` is the **passive** ingest bucket: external
channels push assets into it via the ``upload`` step. This module
provides the side-effect-free helper the step needs:

* ``assemble_day_md`` — render a markdown view of the day's
  ``meta.json`` so the agent (and humans) can browse the bucket as a
  single page without parsing JSON.

Pure: it takes data in, returns data out, touches no disk. The
owning step is responsible for atomic writes and locking.
"""

from __future__ import annotations

from ..schema.resource_meta import ResourceEntry


def assemble_day_md(entries: list[ResourceEntry], date: str, resource_dir: str = "resource") -> str:
    """Render the day's bucket as a markdown view.

    Layout::

        ---
        name: <date>
        assets: [<name1>, <name2>, ...]
        ---

        # <date> resources

        - [[<resource_dir>/<date>/<name>]] — <channel> from `<source>` at <hh:mm> — <description>

    ``resource_dir`` defaults to ``"resource"`` so callers without an
    application context (tests, pure-helper consumers) get the
    conventional layout. The upload step passes the configured
    ``application_config.resource_dir`` so wikilinks always resolve
    against the actual vault layout.

    ``received_at`` is rendered as ``HH:MM`` when it parses as ISO 8601,
    else dropped silently — the asset list stays readable even when
    upstream channels emit malformed timestamps. ``source`` is dropped
    when empty. ``description`` is flattened (newlines collapsed to
    spaces) so the bullet stays one line per asset; ``meta.json``
    preserves the verbatim multi-line text for downstream consumers.
    """
    asset_list = ", ".join(e.name for e in entries) if entries else ""
    lines: list[str] = [
        "---",
        f"name: {date}",
        f"assets: [{asset_list}]",
        "---",
        "",
        f"# {date} resources",
        "",
    ]
    for entry in entries:
        bits: list[str] = [f"- [[{resource_dir}/{date}/{entry.name}]]"]
        provenance = entry.channel
        if entry.source:
            provenance += f" from `{entry.source}`"
        time_part = _hhmm(entry.received_at)
        if time_part:
            provenance += f" at {time_part}"
        bits.append(provenance)
        if entry.description:
            # Flatten so the bullet stays one line per asset; meta.json keeps
            # the verbatim multi-line description for downstream consumers.
            bits.append(" ".join(entry.description.split()))
        lines.append(" — ".join(bits))
    return "\n".join(lines) + "\n"


def _hhmm(received_at: str) -> str:
    """Best-effort HH:MM extraction from an ISO 8601 timestamp."""
    if not received_at:
        return ""
    import datetime as _dt

    raw = received_at.replace("Z", "+00:00")
    try:
        return _dt.datetime.fromisoformat(raw).strftime("%H:%M")
    except ValueError:
        return ""
