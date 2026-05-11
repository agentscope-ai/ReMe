"""Vault path generation + naming disambiguation.

Pure-Python helpers consumed by the `sync` MCP step shell and any
future flow that needs to materialize a path under the vault layout.
No async, no MCP awareness — easy to unit-test.

Wikilink uniqueness is a *graph* property, not a path-builder concern —
see `reme2.component.file_store.BaseFileStore.collisions_after_create`
and `BaseFileStore.all_ambiguous_wikilinks`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date as date_type
from pathlib import Path


def event_path(
    vault_root: str | Path,
    name: str,
    on_date: date_type | str | None = None,
    events_dir: str = "events",
) -> Path:
    """events/{YYYY-MM-DD}/{name}/{name}.md under the given vault root."""
    if on_date is None:
        on_date = date_type.today()
    if isinstance(on_date, date_type):
        on_date = on_date.isoformat()
    return Path(vault_root) / events_dir / on_date / name / f"{name}.md"


def is_folder_topic(path: str | Path) -> bool:
    """True if filename stem == parent directory name (folder note convention)."""
    p = Path(path)
    return p.stem == p.parent.name


def next_suffixed_stem(taken: Iterable[str], base: str) -> str:
    """Lowest unused `<base>-N` (N≥2). Returns `base` itself if not taken.

    Used by `sync` when a same-name-on-same-day collision is detected —
    the suggested suffix is *advisory*; the create still rejects so the
    agent can pick a domain-specific qualifier (e.g. `Apple-Inc` vs
    `Apple-Fruit`) that carries more meaning than a numeric suffix.

    Examples (with taken={"BABA", "BABA-2"}):
        next_suffixed_stem(taken, "BABA") == "BABA-3"
        next_suffixed_stem(taken, "MSFT") == "MSFT"
    """
    taken_set = set(taken)
    if base not in taken_set:
        return base
    pattern = re.compile(rf"^{re.escape(base)}-(\d+)$")
    used: set[int] = set()
    for s in taken_set:
        m = pattern.match(s)
        if m:
            try:
                used.add(int(m.group(1)))
            except ValueError:
                continue
    n = 2
    while n in used:
        n += 1
    return f"{base}-{n}"
