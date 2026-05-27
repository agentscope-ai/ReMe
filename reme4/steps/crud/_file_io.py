"""Shared filesystem helpers for CRUD steps.

Two related concerns, both private to the ``crud`` package:

1. **Generic file IO** — path gating, encoding-aware read/write, output
   truncation (used by every CRUD step that touches the filesystem).
2. **Daily-note helpers** — slug validation + ``daily/<date>.md`` index
   rebuild (used by the ``daily_*`` steps). The day index is a derived
   rollup page auto-managed in marker-delimited sections; user-edited
   manual sections are preserved verbatim across refreshes.
"""

import re
from pathlib import Path
from typing import Iterable

import aiofiles
import aiofiles.os
import frontmatter

from ...constants import DEFAULT_MAX_BYTES, MAX_FILE_READ_BYTES, TRUNCATION_NOTICE_MARKER
from ...utils import get_logger

logger = get_logger()

NON_MD_WARNING = (
    "non-markdown file detected; CRUD operations are recommended on markdown files. "
    "Operating in compatibility mode may carry risks of errors."
)

# Modern text formats — assume UTF-8 by convention.
_STANDARD_TEXT_EXTS = {
    ".md",
    ".py",
    ".js",
    ".ts",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".xml",
    ".log",
    ".conf",
    ".ini",
    ".txt",
    ".sh",
}
# Legacy formats that may use ANSI/GBK on Chinese Windows systems.
_NON_STANDARD_EXTS = {".csv", ".bat", ".cmd", ".reg"}


def resolve_path(vault_path: Path, raw: str) -> tuple[Path | None, str | None]:
    """Resolve a `path=` argument against ``vault_path``.

    Rules:
        - Relative paths are joined under ``vault_path``.
        - Absolute paths are accepted and returned as-is; a warning is logged
          recommending relative paths, but the read still proceeds.
    Returns ``(abs_path, None)`` on success, or ``(None, error_message)`` on failure
    (currently only when ``raw`` is empty/blank).
    Filetype-specific gating (e.g. markdown-only / suffix auto-append) is
    layered on top by callers — see ``reme4/steps/crud/_file_io.py::gate_md``.
    """
    if not raw or not str(raw).strip():
        return None, "`path` is required"
    s = str(raw).strip()
    p = Path(s)
    if p.is_absolute():
        logger.info("absolute path detected, recommending relative paths")
        return p, None
    return vault_path / p, None


def gate_md(target: Path) -> tuple[Path, bool]:
    """Markdown gate with compatibility fallback.

    Returns ``(path, is_md)``:
        - No suffix → auto-append `.md`, ``is_md=True``.
        - `.md` suffix → ``is_md=True``.
        - Any other suffix → ``is_md=False`` (caller handles degraded mode).
    """
    if target.suffix == "":
        return target.with_suffix(".md"), True
    if target.suffix.lower() != ".md":
        return target, False
    return target, True


def _try_decode(data: bytes, encodings: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(text, encoding)`` for the first encoding that decodes ``data`` cleanly."""
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def decode_known_file(data: bytes, file_extension: str) -> tuple[str, str]:
    """Decode file bytes using the extension as a hint. Returns ``(text, encoding)``.

    Strategy:
        1. BOM-based detection.
        2. Extension-driven defaults:
        3. Last resort → UTF-8 with ``errors='replace'`` so the function never raises.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass

    ext = (file_extension or "").lower()

    if ext in _STANDARD_TEXT_EXTS:
        try:
            return data.decode("utf-8-sig"), "utf-8"
        except UnicodeDecodeError:
            pass  # fall through

    if ext in _NON_STANDARD_EXTS:
        result = _try_decode(data, ("utf-8-sig", "gbk"))
        if result is not None:
            text, enc = result
            return text, "utf-8" if enc == "utf-8-sig" else enc

    # Unknown extension or earlier strategies failed.

    return data.decode("utf-8", errors="replace"), "utf-8"


async def read_file_safe(file_path, max_bytes: int = MAX_FILE_READ_BYTES) -> str:
    """Read file in byte mode and decode to string using extension-aware strategy."""
    stat = await aiofiles.os.stat(str(file_path))
    read_size = min(stat.st_size, max_bytes)
    async with aiofiles.open(str(file_path), "rb") as f:
        data = await f.read(read_size)
    text, _ = decode_known_file(data, Path(file_path).suffix)
    return text


async def detect_file_encoding(file_path, sniff_bytes: int = 8192) -> str:
    """Detect the encoding of an existing file so writes can preserve it.

    Reads up to ``sniff_bytes`` from the head of the file (enough for BOM
    detection and statistical analysis). Falls back to ``utf-8`` if the file
    is unreadable.
    """
    try:
        async with aiofiles.open(str(file_path), "rb") as f:
            data = await f.read(sniff_bytes)
    except Exception:  # pylint: disable=broad-except
        return "utf-8"
    _, enc = decode_known_file(data, Path(file_path).suffix)
    return enc


async def write_file_safe(file_path: Path, content: str | bytes, encoding: str = "utf-8") -> None:
    """Write ``content`` to ``file_path`` in binary mode; creates parent dirs.

    ``str`` input is encoded with ``encoding`` (default UTF-8); callers wanting
    to preserve a file's original encoding should pass the result of
    :func:`detect_file_encoding`. If the requested ``encoding`` can't represent
    some characters, falls back to UTF-8 to avoid data loss.

    ``bytes`` input is written verbatim — callers managing their own encoding
    can pass raw bytes directly.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        try:
            payload = content.encode(encoding)
        except (UnicodeEncodeError, LookupError):
            logger.warning(
                "write_file_safe: %r cannot encode all chars, falling back to utf-8",
                encoding,
            )
            payload = content.encode("utf-8")
    else:
        payload = content
    async with aiofiles.open(str(file_path), "wb") as f:
        await f.write(payload)


def truncate_text_output(
    text: str,
    *,
    start_line: int = 1,
    total_lines: int = 0,
    max_bytes: int = DEFAULT_MAX_BYTES,
    file_path: str | None = None,
    encoding: str = "utf-8",
) -> str:
    """Truncate text by bytes preserving line integrity; append a continuation notice.

    See qwenpaw `tools/utils.py` for the same semantics. Returns text unchanged when
    it fits within max_bytes, when max_bytes <= 0, or when the last line itself
    exceeds max_bytes (unhandled edge case).
    """
    if not text or max_bytes <= 0:
        return text

    try:
        text_bytes = text.encode(encoding)
        if len(text_bytes) <= max_bytes:
            return text

        truncated = text_bytes[:max_bytes]
        result = truncated.decode(encoding, errors="ignore")
        newline_count = result.count("\n")
        next_line = start_line + max(1, newline_count)

        if next_line <= total_lines:
            read_from = next_line
        elif start_line < total_lines:
            read_from = total_lines
        else:
            return result

        notice = (
            TRUNCATION_NOTICE_MARKER + f"\nThe output above was truncated."
            f"\nThe full content is saved to the file and contains {total_lines} lines in total."
            f"\nThis excerpt starts at line {start_line} and covers the next {max_bytes} bytes."
            f"\nIf the current content is not enough, call `read` with file={file_path or ''} "
            f"start_line={read_from} to read more."
        )
        return result + notice
    except Exception:
        logger.warning("truncate_text_output failed, returning original text", exc_info=True)
        return text


# ---------------------------------------------------------------------------
# Daily-note helpers: slug validation + day-index rebuild
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


# Day-index rebuild
# -----------------
# The day index is a derived artifact whose single job is daily-note
# consolidation — its source of truth lives in each note's
# frontmatter. The rebuild refreshes auto-managed sections while
# preserving any manual content the user has added between markers.
#
# Frontmatter shape — only the two reserved fields:
#     name:        <date>
#     description: <one-line note-count digest>
#
# The note inventory lives in the body's ``<!-- notes:auto -->``
# wikilinks (graph edges feed off them). Body auto sections (rebuilt
# on every refresh, marker-delimited) — currently just ``notes``.
# Manual sections live outside the auto markers and are preserved
# verbatim across refreshes. A fresh day file gets a ``## 备忘``
# section seeded as the manual scratch area.

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
    happening today" without opening each note."""
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
    """Replace an existing auto block in-place; append at end if absent."""
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
    """Reserved-field-only frontmatter for the index page."""
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

    Returns ``{date, path, notes, created}``.
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
