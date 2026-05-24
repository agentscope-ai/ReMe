"""``_day_index`` — internal helper: build/refresh ``daily/<date>.md`` index page.

The day index is a derived artifact whose single job is **workspace
consolidation** — its source of truth lives in each workspace's
frontmatter. This module rebuilds the auto-managed sections of the
index page while preserving any manual content the user has added
between markers.

Frontmatter shape — only the two reserved fields::

    name:        <date>
    description: <one-line workspace-count digest>

The workspace inventory lives in the body's
``<!-- workspaces:auto -->`` wikilinks (graph edges feed off them).
No bespoke status / lifecycle / scope / role / source / created
axes — those are user-defined and intentionally absent from the
auto-managed payload.

Body auto sections (rebuilt on every refresh, marker-delimited):

* ``workspaces`` — bulleted list of ``[[link]]\\n  name — description`` rows

Manual sections live outside the auto markers and are preserved verbatim
across refreshes. A fresh day file gets a ``## 备忘`` section seeded as
the manual scratch area.

Entry point: ``refresh_day_index(file_store, date)`` — idempotent, safe
to call after every workspace mutation. ``daily_reindex_step`` exposes
it as a standalone tool; orchestrators (synchronizer, batch flows) call
it explicitly after they finish writing.
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter

# Marker syntax: HTML comments so they're invisible in rendered markdown
# but trivially detectable in source. Each block has a paired open/close.
_BLOCK_NAMES = ("workspaces",)
_BLOCK_OPEN = "<!-- {name}:auto -->"
_BLOCK_CLOSE = "<!-- /{name}:auto -->"

_HEADINGS = {
    "workspaces": "## 工作区",
}

_MANUAL_HEADING = "## 备忘"
_MANUAL_STUB = "（人工记录区，刷新索引时不会动）"


def _block_re(name: str) -> re.Pattern:
    """Capturing regex for an auto block: heading + open marker + inner + close."""
    return re.compile(
        rf"(?P<heading>^{re.escape(_HEADINGS[name])}\s*\n)?"
        rf"{re.escape(_BLOCK_OPEN.format(name=name))}"
        r"(?P<inner>.*?)"
        rf"{re.escape(_BLOCK_CLOSE.format(name=name))}",
        re.DOTALL | re.MULTILINE,
    )


def _count_digest(n: int) -> str:
    """One-line workspace count, used as the index ``description``."""
    if n == 0:
        return "本日暂无工作区。"
    return f"今日 {n} 个工作区。"


def _scan_workspaces(vault_dir: Path, date: str, daily_dir: str) -> list[dict]:
    """Walk ``<daily_dir>/<date>/*/`` and pull each workspace's summary frontmatter.

    Returns one dict per workspace::

        {"slug": str, "path": str, "name": str, "description": str}

    Workspaces with no ``<slug>/<slug>.md`` summary are skipped (not a
    valid workspace). Only reserved fields (name / description) are
    read — user-defined frontmatter keys are ignored by the index.
    """
    date_dir = vault_dir / daily_dir / date
    if not date_dir.is_dir():
        return []
    out: list[dict] = []
    for slug_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
        slug = slug_dir.name
        summary = slug_dir / f"{slug}.md"
        if not summary.is_file():
            continue
        try:
            post = frontmatter.loads(summary.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = post.metadata or {}
        out.append(
            {
                "slug": slug,
                "path": f"{daily_dir}/{date}/{slug}/{slug}.md",
                "name": str(meta.get("name") or slug),
                "description": str(meta.get("description") or "").strip(),
            },
        )
    return out


def _render_workspaces_block(workspaces: list[dict]) -> str:
    """Bulleted workspace digest: link on the bullet line, then an
    indented ``name — description`` summary so an agent can scan
    "what's happening today" without opening each workspace.

    The indented summary is omitted entirely when both name and
    description add no information beyond the slug already shown in
    the link.
    """
    if not workspaces:
        return "（无）"
    lines: list[str] = []
    for ws in workspaces:
        lines.append(f"- [[{ws['path']}]]")
        name = ws["name"] if ws["name"] and ws["name"] != ws["slug"] else ""
        description = ws["description"]
        if name and description:
            lines.append(f"  {name} — {description}")
        elif name:
            lines.append(f"  {name}")
        elif description:
            lines.append(f"  {description}")
    return "\n".join(lines)


def _wrap_block(name: str, inner: str) -> str:
    """Wrap rendered inner content with heading + auto markers."""
    return f"{_HEADINGS[name]}\n" f"{_BLOCK_OPEN.format(name=name)}\n" f"{inner}\n" f"{_BLOCK_CLOSE.format(name=name)}"


def _replace_or_append(body: str, name: str, fresh_block: str) -> str:
    """Replace an existing auto block in-place; append at end if absent.

    The replacement keeps the user's heading line if they renamed the
    auto-heading (we only own the marker-wrapped inner). Appending uses
    our canonical heading + markers so future refreshes find them.
    """
    pattern = _block_re(name)
    if pattern.search(body):
        replacement = f"{_BLOCK_OPEN.format(name=name)}\n" f"{fresh_block}\n" f"{_BLOCK_CLOSE.format(name=name)}"
        # Preserve the heading the user had (if any) by only swapping
        # the marker-wrapped portion.
        return pattern.sub(
            lambda m: (m.group("heading") or "") + replacement,
            body,
            count=1,
        )
    # Not present — append the canonical heading + block at the tail.
    suffix = _wrap_block(name, fresh_block)
    return f"{body.rstrip()}\n\n{suffix}\n" if body.strip() else f"{suffix}\n"


def _seed_body(blocks: dict[str, str]) -> str:
    """Fresh-file body: all auto blocks in canonical order + manual stub."""
    parts = [_wrap_block(name, blocks[name]) for name in _BLOCK_NAMES]
    parts.append(f"{_MANUAL_HEADING}\n{_MANUAL_STUB}")
    return "\n\n".join(parts) + "\n"


def _merge_blocks(body: str, blocks: dict[str, str]) -> str:
    """Refresh every auto block in-place; never touch manual content."""
    for name in _BLOCK_NAMES:
        body = _replace_or_append(body, name, blocks[name])
    return body


def _frontmatter_payload(date: str, workspaces: list[dict]) -> dict:
    """Reserved-field-only frontmatter for the index page.

    Emits ``name`` / ``description`` and nothing else — other axes
    (status / lifecycle / scope / role / source / created) are
    user-defined and belong in workspace summaries, not in this
    derived aggregate.
    """
    return {
        "name": date,
        "description": _count_digest(len(workspaces)),
    }


async def refresh_day_index(file_store, date: str, daily_dir: str = "daily") -> dict:
    """Rebuild ``<daily_dir>/<date>.md`` from the current state of its workspaces.

    Behaviour:
    * No ``<daily_dir>/<date>/`` at all and no existing index file → no-op.
    * Workspaces present → write the index file (create if missing,
      otherwise merge auto blocks into the existing body, preserve
      manual segments, refresh frontmatter).
    * Workspaces directory empty but index file exists → rebuild with
      empty auto blocks (keeps the file in sync with reality).

    ``daily_dir`` defaults to ``"daily"`` for tests / pure-helper
    consumers; the registered steps pass the configured
    ``application_config.daily_dir`` so the on-disk layout always
    matches what the index file claims.

    Returns::

        {
          "date": str,
          "path": "<daily_dir>/<date>.md",
          "workspaces": [
              {"path": "<daily_dir>/<date>/<slug>/<slug>.md",
               "name": str,
               "description": str},
              ...
          ],
          "created": bool,   # True if index file was just written for the first time
        }

    The ``workspaces`` list mirrors the order rendered in the index body
    (sorted by slug). The ``created`` field reflects index-page creation,
    not workspace creation, so callers can log "index emerged" events
    distinctly.
    """
    vault_dir = Path(file_store.vault_path or ".").resolve()
    index_rel = f"{daily_dir}/{date}.md"
    index_abs = vault_dir / index_rel
    workspaces = _scan_workspaces(vault_dir, date, daily_dir)

    workspace_payload = [
        {"path": ws["path"], "name": ws["name"], "description": ws["description"]} for ws in workspaces
    ]

    # Nothing to index and no prior index file — quietly do nothing.
    if not workspaces and not index_abs.is_file():
        return {
            "date": date,
            "path": index_rel,
            "workspaces": workspace_payload,
            "created": False,
        }

    blocks = {"workspaces": _render_workspaces_block(workspaces)}

    if index_abs.is_file():
        post = frontmatter.loads(index_abs.read_text(encoding="utf-8"))
        new_body = _merge_blocks(post.content, blocks)
        was_created = False
    else:
        index_abs.parent.mkdir(parents=True, exist_ok=True)
        new_body = _seed_body(blocks)
        was_created = True

    fm = _frontmatter_payload(date, workspaces)
    out = frontmatter.Post(new_body, **fm)
    index_abs.write_text(frontmatter.dumps(out), encoding="utf-8")

    return {
        "date": date,
        "path": index_rel,
        "workspaces": workspace_payload,
        "created": was_created,
    }
