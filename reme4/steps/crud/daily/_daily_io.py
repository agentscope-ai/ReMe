"""Internal helpers for daily-aware steps — slug validation + day-index rebuild.

Two related concerns, both private to the ``daily`` package:

1. **Slug naming** — Windows-safe filename validation for the slug
   that becomes the stem of ``daily/<YYYY-MM-DD>/<slug>.md``.
2. **Day-index** — the derived rollup page ``daily/<YYYY-MM-DD>.md``
   listing every note under that date with name + description. The
   index is auto-managed in marker-delimited sections; user-edited
   manual sections are preserved verbatim across refreshes.

Public entry points:

* :func:`validate_slug` — return an error string, or ``None`` when the
  slug is safe to use as a filename.
* :func:`scan_notes` — walk ``<daily_dir>/<date>/*.md`` and pull each
  note's reserved frontmatter (``name`` / ``description``).
* :func:`refresh_day_index` — rebuild ``<daily_dir>/<date>.md`` from
  the current state of its notes. Idempotent, safe to re-run.
"""

import re
from pathlib import Path

import frontmatter

# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_slug(slug: str) -> str | None:
    """Return an error message, or ``None`` when ``slug`` is a safe filename.

    Rules (Windows is the strictest filesystem, so we validate to its bar):

    - non-empty, no leading / trailing whitespace
    - no reserved characters: ``< > : " / \\ | ? *`` or control chars (``\\x00-\\x1f``)
    - no reserved device names: ``CON`` / ``PRN`` / ``AUX`` / ``NUL`` /
      ``COM1-9`` / ``LPT1-9`` (Windows reserves these with or without an
      extension — ``CON.txt`` is also forbidden)
    - no trailing ``.``
    """
    if not slug:
        return "slug is required"
    if slug != slug.strip():
        return f"slug cannot have leading or trailing whitespace: {slug!r}"
    if _INVALID_CHARS.search(slug):
        return f'slug contains invalid characters (one of < > : " / \\ | ? * ' f"or a control char): {slug!r}"
    if slug.endswith("."):
        return f"slug cannot end with '.': {slug!r}"
    if slug.split(".", 1)[0].upper() in _RESERVED_NAMES:
        return f"slug is a Windows-reserved device name: {slug!r}"
    return None


# ---------------------------------------------------------------------------
# Day-index rebuild
# ---------------------------------------------------------------------------
#
# The day index is a derived artifact whose single job is **daily-note
# consolidation** — its source of truth lives in each note's
# frontmatter. The rebuild refreshes auto-managed sections while
# preserving any manual content the user has added between markers.
#
# Frontmatter shape — only the two reserved fields::
#
#     name:        <date>
#     description: <one-line note-count digest>
#
# The note inventory lives in the body's ``<!-- notes:auto -->``
# wikilinks (graph edges feed off them). No bespoke status / lifecycle
# / scope / role / source / created axes — those are user-defined and
# intentionally absent from the auto-managed payload.
#
# Body auto sections (rebuilt on every refresh, marker-delimited):
#
# * ``notes`` — bulleted list of ``[[link]]\n  name — description`` rows
#
# Manual sections live outside the auto markers and are preserved
# verbatim across refreshes. A fresh day file gets a ``## 备忘``
# section seeded as the manual scratch area.

# Marker syntax: HTML comments so they're invisible in rendered markdown
# but trivially detectable in source. Each block has a paired open/close.
_BLOCK_NAMES = ("notes",)
_BLOCK_OPEN = "<!-- {name}:auto -->"
_BLOCK_CLOSE = "<!-- /{name}:auto -->"

_HEADINGS = {
    "notes": "## 今日笔记",
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
    """One-line note count, used as the index ``description``."""
    if n == 0:
        return "本日暂无笔记。"
    return f"今日 {n} 篇笔记。"


def scan_notes(vault_dir: Path, date: str, daily_dir: str) -> list[dict]:
    """Walk ``<daily_dir>/<date>/*.md`` and pull each note's frontmatter.

    Returns one dict per note::

        {"slug": str, "path": str, "name": str, "description": str}

    Each ``.md`` directly under the day folder is a note; the file's
    stem is the slug. Only reserved fields (name / description) are
    read — user-defined frontmatter keys are ignored by the index.
    """
    date_dir = vault_dir / daily_dir / date
    if not date_dir.is_dir():
        return []
    out: list[dict] = []
    for md_path in sorted(p for p in date_dir.iterdir() if p.is_file() and p.suffix == ".md"):
        slug = md_path.stem
        try:
            post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
        except Exception:  # pylint: disable=broad-except
            continue
        meta = post.metadata or {}
        out.append(
            {
                "slug": slug,
                "path": f"{daily_dir}/{date}/{slug}.md",
                "name": str(meta.get("name") or slug),
                "description": str(meta.get("description") or "").strip(),
            },
        )
    return out


def _render_notes_block(notes: list[dict]) -> str:
    """Bulleted note digest: link on the bullet line, then an indented
    ``name — description`` summary so an agent can scan "what's
    happening today" without opening each note.

    The indented summary is omitted entirely when both name and
    description add no information beyond the slug already shown in
    the link.
    """
    if not notes:
        return "（无）"
    lines: list[str] = []
    for note in notes:
        lines.append(f"- [[{note['path']}]]")
        name = note["name"] if note["name"] and note["name"] != note["slug"] else ""
        description = note["description"]
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
        return pattern.sub(
            lambda m: (m.group("heading") or "") + replacement,
            body,
            count=1,
        )
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


def _frontmatter_payload(date: str, notes: list[dict]) -> dict:
    """Reserved-field-only frontmatter for the index page.

    Emits ``name`` / ``description`` and nothing else — other axes
    (status / lifecycle / scope / role / source / created) are
    user-defined and belong in note bodies, not in this derived
    aggregate.
    """
    return {
        "name": date,
        "description": _count_digest(len(notes)),
    }


async def refresh_day_index(file_store, date: str, daily_dir: str = "daily") -> dict:
    """Rebuild ``<daily_dir>/<date>.md`` from the current state of its notes.

    Behaviour:
    * No ``<daily_dir>/<date>/`` at all and no existing index file → no-op.
    * Notes present → write the index file (create if missing,
      otherwise merge auto blocks into the existing body, preserve
      manual segments, refresh frontmatter).
    * Notes directory empty but index file exists → rebuild with
      empty auto blocks (keeps the file in sync with reality).

    ``daily_dir`` defaults to ``"daily"`` for tests / pure-helper
    consumers; the registered steps pass the configured
    ``application_config.daily_dir`` so the on-disk layout always
    matches what the index file claims.

    Returns::

        {
          "date": str,
          "path": "<daily_dir>/<date>.md",
          "notes": [
              {"path": "<daily_dir>/<date>/<slug>.md",
               "name": str,
               "description": str},
              ...
          ],
          "created": bool,   # True if index file was just written for the first time
        }

    The ``notes`` list mirrors the order rendered in the index body
    (sorted by slug). The ``created`` field reflects index-page creation,
    not note creation, so callers can log "index emerged" events
    distinctly.
    """
    vault_dir = Path(file_store.vault_path or ".").resolve()
    index_rel = f"{daily_dir}/{date}.md"
    index_abs = vault_dir / index_rel
    notes = scan_notes(vault_dir, date, daily_dir)

    notes_payload = [{"path": n["path"], "name": n["name"], "description": n["description"]} for n in notes]

    if not notes and not index_abs.is_file():
        return {
            "date": date,
            "path": index_rel,
            "notes": notes_payload,
            "created": False,
        }

    blocks = {"notes": _render_notes_block(notes)}

    if index_abs.is_file():
        post = frontmatter.loads(index_abs.read_text(encoding="utf-8"))
        new_body = _merge_blocks(post.content, blocks)
        was_created = False
    else:
        index_abs.parent.mkdir(parents=True, exist_ok=True)
        new_body = _seed_body(blocks)
        was_created = True

    fm = _frontmatter_payload(date, notes)
    out = frontmatter.Post(new_body, **fm)
    index_abs.write_text(frontmatter.dumps(out), encoding="utf-8")

    return {
        "date": date,
        "path": index_rel,
        "notes": notes_payload,
        "created": was_created,
    }
