"""Path resolution, validation, and filetype gating for CRUD steps."""

import re
from pathlib import Path

from ...utils import get_logger

logger = get_logger()

NON_MD_WARNING = (
    "non-markdown file detected; CRUD operations are recommended on markdown files. "
    "Operating in compatibility mode may carry risks of errors."
)

NON_IMAGE_WARNING = (
    "non-image file detected; CRUD image operations are recommended on standard image formats. "
    "Operating in compatibility mode may carry risks of errors."
)

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
    """Return an error message, or ``None`` when ``name`` is a safe filename component."""
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

    Returns ``(abs_path, None)`` on success, or ``(None, error_message)`` on failure.
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
        - No suffix -> auto-append `.md`, ``is_md=True``.
        - `.md` suffix -> ``is_md=True``.
        - Any other suffix -> ``is_md=False`` (caller handles degraded mode).
    """
    if target.suffix == "":
        return target.with_suffix(".md"), True
    if target.suffix.lower() != ".md":
        return target, False
    return target, True


def gate_image(target: Path) -> tuple[Path, bool, str | None]:
    """Image gate with compatibility fallback. Returns ``(path, is_image, mime)``."""
    mime = IMAGE_MIME_BY_EXT.get(target.suffix.lower())
    return target, mime is not None, mime
