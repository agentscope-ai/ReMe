"""Persist Auto Memory image attachments and retain their note references."""

import base64
import binascii
from pathlib import Path
from urllib.parse import urlsplit

from agentscope.message import DataBlock, Msg, TextBlock
import frontmatter

from ..file_io._file_io import get_path_lock, write_file_safe
from ..file_io._path import _check_path_permission, resolve_path, validate_filename_component
from ...utils.wikilink_handler import WikilinkHandler

_IMAGE_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/avif": ".avif",
    "image/svg+xml": ".svg",
}


def _checked_path(workspace: Path, relative: str, allowed_paths) -> Path:
    target, error = resolve_path(workspace, relative)
    if error or target is None:
        raise ValueError(error or "Invalid image path")
    if not _check_path_permission(workspace, target, allowed_paths):
        raise PermissionError(f"No permission to write {relative}")
    return target


def _check_existing_image(target: Path, payload: bytes) -> bool:
    if not target.exists():
        return False
    if not target.is_file() or target.read_bytes() != payload:
        raise ValueError(f"Image attachment already exists with different content: {target.name}")
    return True


async def save_session_images(
    workspace: Path,
    session_dir: str,
    session_id: str,
    messages: list[Msg],
    images: dict[str, DataBlock],
    allowed_paths=None,
) -> list[str]:
    """Save Base64 bytes and annotate prepared image markers; URLs remain remote.

    ``messages`` must be the invocation-owned copies from image preparation.
    Image data and caller-owned messages remain unchanged. Image identity uses
    the message ID and block position, not the image block's optional identity.
    """
    workspace = workspace.resolve()
    if error := validate_filename_component(session_id, kind="session_id"):
        raise ValueError(error)
    if Path(session_dir).is_absolute():
        raise ValueError("session_dir must be workspace-relative")
    pending: dict[Path, tuple[str, bytes]] = {}
    replacements = []
    sources = []
    for message in messages:
        for index, block in enumerate(message.content):
            if not isinstance(block, TextBlock) or block.text not in images:
                continue
            image = images[block.text]
            source = image.source
            if source.type == "url":
                reference = str(source.url)
                if urlsplit(reference).scheme not in {"http", "https"}:
                    raise ValueError("Image URLs must use HTTP(S)")
            else:
                try:
                    payload = base64.b64decode(source.data, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("Image source contains invalid Base64") from exc
                if not payload:
                    raise ValueError("Image source contains empty Base64 data")
                # A reversible encoding, not a hash: also distinguish IDs on
                # case-insensitive filesystems without trusting IDs as paths.
                encoded_id = message.id.encode("utf-8").hex()
                suffix = _IMAGE_EXTENSIONS.get(source.media_type.lower(), ".bin")
                filename = f"msg-{encoded_id}-image-{index}{suffix}"
                if len(filename) > 255:
                    raise ValueError("Message ID is too long for an image attachment filename")
                relative = (Path(session_dir) / "images" / session_id / filename).as_posix()
                if error := WikilinkHandler.validate_src_dst(relative, relative):
                    raise ValueError(f"Image attachment path cannot be linked: {error}")
                target = _checked_path(workspace, relative, allowed_paths)
                if target in pending and pending[target][1] != payload:
                    raise ValueError("Conflicting images share the same message ID and block position")
                _check_existing_image(target, payload)
                pending[target] = (relative, payload)
                reference = f"[[{relative}]]"
            replacements.append((message, index, block, reference))
            if reference not in sources:
                sources.append(reference)

    # Validate every input before creating files. Recheck each path under the
    # existing file-operation lock so concurrent saves cannot overwrite it.
    for relative, payload in pending.values():
        target = _checked_path(workspace, relative, allowed_paths)
        async with await get_path_lock(target):
            target = _checked_path(workspace, relative, allowed_paths)
            if not _check_existing_image(target, payload):
                await write_file_safe(target, payload)
    for message, index, block, reference in replacements:
        message.content[index] = block.model_copy(update={"text": f"Image source: {reference}\n{block.text}"})
    return sources


async def merge_image_sources(
    workspace: Path,
    note_path: str,
    sources: list[str],
    allowed_paths=None,
) -> None:
    """Merge source links with the latest frontmatter inside the file's lock."""
    if not isinstance(sources, list) or any(not isinstance(source, str) for source in sources):
        raise ValueError("source_images must be a list of strings")
    if not sources:
        return
    workspace = workspace.resolve()
    target = _checked_path(workspace, note_path, allowed_paths)
    async with await get_path_lock(target):
        target = _checked_path(workspace, note_path, allowed_paths)
        if not target.is_file():
            raise ValueError(f"Memory note not found: {note_path}")
        post = frontmatter.loads(target.read_text(encoding="utf-8"))
        current = post.metadata.get("source_images", [])
        if not isinstance(current, list) or any(not isinstance(source, str) for source in current):
            raise ValueError("Existing source_images must be a list of strings")
        merged = list(dict.fromkeys([*current, *sources]))
        if current == merged:
            return
        post.metadata["source_images"] = merged
        await write_file_safe(target, frontmatter.dumps(post))
