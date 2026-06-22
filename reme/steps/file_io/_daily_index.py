"""Daily-note helpers: session_id validation + day-index rebuild."""

from pathlib import Path

import frontmatter

from ._path import resolve_path, validate_filename_component
from ...utils import get_logger

logger = get_logger()


def validate_session_id(session_id: str) -> str | None:
    """Validate a daily-note session_id. Thin wrapper over :func:`validate_filename_component`."""
    return validate_filename_component(session_id, kind="session_id")


_NOTES_OPEN = "<!-- notes:auto -->"
_NOTES_CLOSE = "<!-- /notes:auto -->"


def _render_notes_block(notes: list[dict]) -> str:
    """Render each note as ``- [[path]] key: val ...`` (one line per note)."""
    if not notes:
        return "(none)"
    lines: list[str] = []
    for note in notes:
        meta: dict = note["metadata"]
        keys = [k for k in ("name", "description") if k in meta] + [k for k in meta if k not in ("name", "description")]
        parts = [f"- [[{note['path']}]]"] + [
            f"{k}: {str(v).replace(chr(10), ' ')}" for k in keys if (v := meta[k]) not in (None, "")
        ]
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _rebuild_body(body: str, notes_content: str) -> str:
    """Replace or append the auto block, preserving surrounding content."""
    block = f"{_NOTES_OPEN}\n{notes_content}\n{_NOTES_CLOSE}"
    if _NOTES_OPEN in body and _NOTES_CLOSE in body:
        return body.split(_NOTES_OPEN, 1)[0] + block + body.split(_NOTES_CLOSE, 1)[1]
    return f"{body.rstrip()}\n\n{block}\n" if body.strip() else f"{block}\n"


def scan_notes(workspace_dir: Path, date: str, daily_dir: str) -> list[dict]:
    """Walk ``<daily_dir>/<date>/*.md`` and pull each note's frontmatter.

    Returns one dict per note::

        {"session_id": str, "path": str, "metadata": dict}
    """
    date_dir, err = resolve_path(workspace_dir, f"{daily_dir}/{date}")
    if err or date_dir is None:
        logger.info(f"scan_notes skipped invalid daily path date={date!r} daily_dir={daily_dir!r} error={err!r}")
        return []
    if not date_dir.is_dir():
        return []
    out: list[dict] = []
    for md_path in sorted(p for p in date_dir.iterdir() if p.is_file() and p.suffix == ".md"):
        session_id = md_path.stem
        try:
            post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(
            {
                "session_id": session_id,
                "path": f"{daily_dir}/{date}/{md_path.name}",
                "metadata": dict(post.metadata or {}),
            },
        )
    return out


async def refresh_day_index(file_store, date: str, daily_dir: str) -> dict:
    """Rebuild ``<daily_dir>/<date>.md`` from the current state of its notes.

    Returns ``{date, path, notes, created}``.
    """
    workspace_dir = Path(file_store.workspace_path or ".").resolve()
    index_rel = f"{daily_dir}/{date}.md"
    index_abs, err = resolve_path(workspace_dir, index_rel)
    if err or index_abs is None:
        return {
            "date": date,
            "path": index_rel,
            "notes": [],
            "created": False,
            "changed": False,
            "error": err or "invalid path",
        }
    notes = scan_notes(workspace_dir, date, daily_dir)

    notes_payload = [{"path": n["path"], "session_id": n["session_id"], "metadata": n["metadata"]} for n in notes]

    if not notes and not index_abs.is_file():
        return {
            "date": date,
            "path": index_rel,
            "notes": notes_payload,
            "created": False,
            "changed": False,
        }

    notes_block = _render_notes_block(notes)

    n = len(notes)
    fm = {"name": date, "description": "No notes today." if n == 0 else f"{n} note(s) today."}

    existing_text = ""
    if index_abs.is_file():
        existing_text = index_abs.read_text(encoding="utf-8")
        post = frontmatter.loads(existing_text)
        new_body = _rebuild_body(post.content, notes_block)
        merged = dict(post.metadata or {})
        if not merged.get("name"):
            merged["name"] = fm["name"]
        merged["description"] = fm["description"]
        fm = merged
        was_created = False
    else:
        index_abs.parent.mkdir(parents=True, exist_ok=True)
        new_body = f"{_NOTES_OPEN}\n{notes_block}\n{_NOTES_CLOSE}\n"
        was_created = True
    out = frontmatter.Post(new_body, **fm)
    rendered = frontmatter.dumps(out)
    changed = not index_abs.is_file() or existing_text != rendered
    if changed:
        index_abs.write_text(rendered, encoding="utf-8")

    return {
        "date": date,
        "path": index_rel,
        "notes": notes_payload,
        "created": was_created,
        "changed": changed,
    }
