"""Shared filesystem helpers for CRUD steps.

Two related concerns, both private to the ``crud`` package:

1. **Generic file IO** — path gating, encoding-aware read/write, output
   truncation (used by every CRUD step that touches the filesystem).
2. **Daily-note helpers** — slug validation + ``daily/<date>.md`` index
   rebuild (used by the ``daily_*`` steps). The day index is a derived
   rollup page auto-managed in marker-delimited sections; user-edited
   manual sections are preserved verbatim across refreshes.
"""

import asyncio
import re
from pathlib import Path
from typing import Iterable

import aiofiles
import aiofiles.os
import frontmatter

from ...constants import DEFAULT_MAX_BYTES, MAX_FILE_READ_BYTES, TRUNCATION_NOTICE_MARKER
from ...utils import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Generic file IO
# ---------------------------------------------------------------------------

NON_MD_WARNING = (
    "non-markdown file detected; CRUD operations are recommended on markdown files. "
    "Operating in compatibility mode may carry risks of errors."
)

NON_IMAGE_WARNING = (
    "non-image file detected; CRUD image operations are recommended on standard image formats. "
    "Operating in compatibility mode may carry risks of errors."
)

# Image suffix → MIME mapping. SVG intentionally excluded (text format, base64
# encoding has no benefit — caller should use read_step instead).
IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
}

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


# ---------------------------------------------------------------------------
# In-process per-path write lock.
#
# Concurrent CRUD writes (write / edit) targeting the same path from the
# same process must be serialized so a read-modify-write cycle isn't
# interleaved by another coroutine. Different paths get different locks,
# so unrelated writes still run in parallel.
#
# NOTE: this is in-process only — multi-worker / multi-process deployments
# are NOT protected. That trade-off is acceptable for the current single-
# process reme server; cross-process protection would need flock or OCC.
# ---------------------------------------------------------------------------
_PATH_LOCKS: dict[str, asyncio.Lock] = {}
_PATH_LOCKS_REGISTRY = asyncio.Lock()


async def get_path_lock(target: Path) -> asyncio.Lock:
    """Return the asyncio.Lock for ``target``; created lazily on first request.

    The lock is keyed by the string form of ``target`` — callers should pass
    a path that has already been normalized by :func:`resolve_path` so two
    equivalent paths share one lock.
    """
    key = str(target)
    async with _PATH_LOCKS_REGISTRY:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _PATH_LOCKS[key] = lock
    return lock


# Path helpers
# ------------

# Filename validation. Windows is the strictest mainstream filesystem, so we
# validate to its bar — paths that pass here also work on macOS and Linux,
# and survive sync to a Windows machine or zip-and-share workflows.

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_filename_component(name: str, *, kind: str = "filename") -> str | None:
    """Return an error message, or ``None`` when ``name`` is a safe filename component.

    A *component* is a single path segment — no ``/`` or ``\\`` allowed inside.
    Used for both daily-note slugs and ``resolve_path`` per-component checks.

    Rules:

    - non-empty, no leading / trailing whitespace
    - no reserved characters: ``< > : " / \\ | ? *`` or control chars (``\\x00-\\x1f``)
    - no reserved device names: ``CON`` / ``PRN`` / ``AUX`` / ``NUL`` /
      ``COM1-9`` / ``LPT1-9`` (Windows reserves these with or without an
      extension — ``CON.txt`` is also forbidden)
    - no trailing ``.`` (also rejects ``..``, which doubles as path-traversal protection
      for callers that validate per component)

    ``kind`` is the human-readable label inserted into error messages
    (e.g. ``"slug"``, ``"path component"``).
    """
    if not name:
        return f"{kind} is required"
    if name != name.strip():
        return f"{kind} cannot have leading or trailing whitespace: {name!r}"
    if _INVALID_CHARS.search(name):
        return f'{kind} contains invalid characters (one of < > : " / \\ | ? * or a control char): {name!r}'
    if name.endswith("."):
        return f"{kind} cannot end with '.': {name!r}"
    if name.split(".", 1)[0].upper() in _RESERVED_NAMES:
        return f"{kind} is a Windows-reserved device name: {name!r}"
    return None


def resolve_path(vault_path: Path, raw: str) -> tuple[Path | None, str | None]:
    """Resolve a `path=` argument against ``vault_path``.

    Rules:
        - Relative paths are joined under ``vault_path``.
        - Absolute paths are accepted and returned as-is; a warning is logged
          recommending relative paths, but the read still proceeds.
        - Each path component is validated against the same Windows-strict
          filename rules used for daily-note slugs — see
          :func:`validate_filename_component`. ``..`` is rejected by the
          trailing-``.`` rule, which doubles as path-traversal protection.
    Returns ``(abs_path, None)`` on success, or ``(None, error_message)`` on failure.
    Filetype-specific gating (e.g. markdown-only / suffix auto-append) is
    layered on top by callers — see ``reme/steps/file_io/_file_io.py::gate_md``.
    """
    if not raw or not str(raw).strip():
        return None, "`path` is required"
    s = str(raw).strip()
    p = Path(s)
    for part in p.parts:
        if part == p.anchor:
            continue
        err = validate_filename_component(part, kind="path component")
        if err:
            return None, err
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


def gate_image(target: Path) -> tuple[Path, bool, str | None]:
    """Image gate with compatibility fallback. Returns ``(path, is_image, mime)``.

    Behavior diverges from :func:`gate_md` in two ways:
        - **No suffix is NOT auto-appended.** Image formats have no single
          reasonable default; guessing would mislead.
        - **No path mutation** — ``target`` is returned unchanged.

    Suffix routing:
        - Known image suffix (see ``IMAGE_MIME_BY_EXT``) → ``(target, True, mime)``
        - Empty suffix or unknown suffix → ``(target, False, None)``
          Caller may still read the file and surface a ``NON_IMAGE_WARNING``.
    """
    mime = IMAGE_MIME_BY_EXT.get(target.suffix.lower())
    return target, mime is not None, mime


# Encoding detection (private)
# ----------------------------


def _try_decode(data: bytes, encodings: Iterable[str]) -> tuple[str, str] | None:
    """Return ``(text, encoding)`` for the first encoding that decodes ``data`` cleanly."""
    for enc in encodings:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            continue
    return None


def _decode_known_file(data: bytes, file_extension: str) -> tuple[str, str]:
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


# File read / write
# -----------------


async def read_file_safe(file_path, max_bytes: int = MAX_FILE_READ_BYTES) -> tuple[str, str]:
    """Read file in byte mode and decode using extension-aware strategy.

    Returns ``(text, encoding)``. Callers that need to write the file back
    in its original encoding can pass ``encoding`` straight to
    :func:`write_file_safe`, avoiding a second read via
    :func:`detect_file_encoding`.
    """
    stat = await aiofiles.os.stat(str(file_path))
    read_size = min(stat.st_size, max_bytes)
    async with aiofiles.open(str(file_path), "rb") as f:
        data = await f.read(read_size)
    return _decode_known_file(data, Path(file_path).suffix)


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
    _, enc = _decode_known_file(data, Path(file_path).suffix)
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


# Output formatting
# -----------------


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

# Slug validation
# ---------------


def validate_slug(slug: str) -> str | None:
    """Validate a daily-note slug. Thin wrapper over :func:`validate_filename_component`."""
    return validate_filename_component(slug, kind="slug")


# Day-index rebuild
# -----------------
# The day index is a derived artifact whose single job is daily-note
# consolidation — its source of truth lives in each note's
# frontmatter. The rebuild refreshes the auto-managed notes block
# while preserving any user content sitting outside the markers.
#
# Frontmatter shape — only the two reserved fields:
#     name:        <date>
#     description: <one-line note-count digest>
#
# The note inventory lives in the body's ``<!-- notes:auto -->``
# block: each note becomes a single line with its full frontmatter
# inlined (``- [[path]] name: ... description: ... <other keys>``),
# letting an agent scan the day at a glance. Content outside the
# auto markers is preserved verbatim across refreshes.

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


# Public scan + rebuild
# ---------------------


def scan_notes(vault_dir: Path, date: str, daily_dir: str) -> list[dict]:
    """Walk ``<daily_dir>/<date>/*.md`` and pull each note's frontmatter.

    Returns one dict per note::

        {"slug": str, "path": str, "metadata": dict}

    ``metadata`` is the raw frontmatter dict (insertion-ordered);
    consumers decide which keys to surface. Each ``.md`` directly
    under the day folder is a note; the file's stem is the slug.
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
        out.append(
            {
                "slug": slug,
                "path": f"{daily_dir}/{date}/{slug}.md",
                "metadata": dict(post.metadata or {}),
            },
        )
    return out


async def refresh_day_index(file_store, date: str, daily_dir: str) -> dict:
    """Rebuild ``<daily_dir>/<date>.md`` from the current state of its notes.

    Behaviour:
    * No ``<daily_dir>/<date>/`` at all and no existing index file → no-op.
    * Notes present → write the index file (create if missing,
      otherwise refresh the auto block in place, preserving content
      outside the markers, refresh frontmatter).
    * Notes directory empty but index file exists → rebuild with
      empty auto block (keeps the file in sync with reality).

    Returns ``{date, path, notes, created}``. Each row in ``notes`` is
    ``{path, slug, metadata}`` with the raw frontmatter dict.
    """
    vault_dir = Path(file_store.vault_path or ".").resolve()
    index_rel = f"{daily_dir}/{date}.md"
    index_abs = vault_dir / index_rel
    notes = scan_notes(vault_dir, date, daily_dir)

    notes_payload = [{"path": n["path"], "slug": n["slug"], "metadata": n["metadata"]} for n in notes]

    if not notes and not index_abs.is_file():
        return {
            "date": date,
            "path": index_rel,
            "notes": notes_payload,
            "created": False,
        }

    notes_block = _render_notes_block(notes)

    n = len(notes)
    fm = {"name": date, "description": "No notes today." if n == 0 else f"{n} note(s) today."}

    if index_abs.is_file():
        post = frontmatter.loads(index_abs.read_text(encoding="utf-8"))
        new_body = _rebuild_body(post.content, notes_block)
        merged = dict(post.metadata or {})
        for key, value in fm.items():
            if not merged.get(key):
                merged[key] = value
        fm = merged
        was_created = False
    else:
        index_abs.parent.mkdir(parents=True, exist_ok=True)
        new_body = f"{_NOTES_OPEN}\n{notes_block}\n{_NOTES_CLOSE}\n"
        was_created = True
    out = frontmatter.Post(new_body, **fm)
    index_abs.write_text(frontmatter.dumps(out), encoding="utf-8")

    return {
        "date": date,
        "path": index_rel,
        "notes": notes_payload,
        "created": was_created,
    }
