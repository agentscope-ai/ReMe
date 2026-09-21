"""Image resource processor for the unified auto-resource router."""

import base64
import io
import json
import warnings
from pathlib import Path, PurePosixPath

import aiofiles
import frontmatter
from agentscope.agent import ContextConfig
from agentscope.message import Base64Source, DataBlock

from ..file_io._path import IMAGE_SUFFIXES
from .base_auto_resource import BaseAutoResourceStep
from ...components import R

DEFAULT_MAX_IMAGE_INPUT_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
MAX_IMAGE_REQUEST_DIMENSION = 2048
_JPEG_QUALITY = 85
# Decoded formats outside this set are re-encoded to provider-friendly
# PNG/JPEG for VLM requests; the stored resource file is never modified.
_PASSTHROUGH_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_HEIF_BRANDS = frozenset(
    {b"heic", b"heif", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs", b"mif1", b"msf1"},
)
_MAX_FTYP_SCAN_BYTES = 4096


def _load_pillow():
    """Load the core image dependency only when image processing runs."""
    try:
        from PIL import Image, ImageOps  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("Image captioning requires Pillow; install reme-ai[core]") from exc
    return Image, ImageOps


def _looks_like_heif(data: bytes) -> bool:
    """Return whether an ISO-BMFF header declares a HEIC/HEIF brand."""
    if len(data) < 12 or data[4:8] != b"ftyp":
        return False
    box_size = int.from_bytes(data[:4], "big")
    if box_size < 12:
        return False
    end = min(box_size, len(data), _MAX_FTYP_SCAN_BYTES)
    if data[8:12] in _HEIF_BRANDS:
        return True
    return any(data[index : index + 4] in _HEIF_BRANDS for index in range(16, end - 3, 4))


def _register_heif_opener() -> None:
    """Load and register HEIC support only for bytes that declare HEIF."""
    try:
        from pillow_heif import register_heif_opener  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("HEIC image captioning requires pillow-heif; install reme-ai[image-heif]") from exc
    try:
        register_heif_opener()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to initialize HEIC image support: {exc}") from exc


def _normalize_image_bytes(
    data: bytes,
    suffix: str,
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> tuple[bytes | None, str, str]:
    """Validate and optionally normalize image bytes for a VLM request.

    The returned tuple is ``(normalized_bytes, request_mime, source_mime)``.
    ``normalized_bytes`` is ``None`` only when the original bytes can be sent
    unchanged. MIME values come from the decoded image rather than its suffix.
    Missing dependencies, unsafe pixel counts, and decode/convert failures are
    explicit. The stored resource file is never modified.
    """
    try:
        pixel_limit = int(max_image_pixels)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_image_pixels must be a positive integer: {max_image_pixels!r}") from exc
    if pixel_limit <= 0:
        raise ValueError(f"max_image_pixels must be a positive integer: {max_image_pixels!r}")

    image_module, image_ops = _load_pillow()
    if _looks_like_heif(data):
        _register_heif_opener()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", image_module.DecompressionBombWarning)
            image = image_module.open(io.BytesIO(data))
    except (image_module.DecompressionBombWarning, image_module.DecompressionBombError) as exc:
        raise RuntimeError(
            f"Image rejected by Pillow decompression-bomb protection ({suffix or 'unknown suffix'})",
        ) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to decode image ({suffix or 'unknown suffix'}): {exc}") from exc

    with image:
        width, height = image.size
        pixel_count = width * height
        if pixel_count > pixel_limit:
            raise RuntimeError(
                f"Image exceeds max_image_pixels before decode: " f"{width}x{height}={pixel_count} > {pixel_limit}",
            )

        source_mime = str(image.get_format_mimetype() or "").strip().lower()
        if not source_mime.startswith("image/"):
            raise RuntimeError(
                f"Cannot determine decoded image MIME type ({suffix or 'unknown suffix'}, format={image.format!r})",
            )

        needs_resize = width > MAX_IMAGE_REQUEST_DIMENSION or height > MAX_IMAGE_REQUEST_DIMENSION
        if source_mime == "image/jpeg" and needs_resize:
            max_dimension = max(width, height)
            decoder_size = (
                max(1, (width * MAX_IMAGE_REQUEST_DIMENSION + max_dimension - 1) // max_dimension),
                max(1, (height * MAX_IMAGE_REQUEST_DIMENSION + max_dimension - 1) // max_dimension),
            )
            try:
                # JPEG supports power-of-two decoder scaling. ``draft`` picks
                # the smallest decoded frame that still covers decoder_size,
                # reducing peak memory before the final LANCZOS thumbnail.
                image.draft(None, decoder_size)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"Failed to prepare JPEG decoder downsampling ({width}x{height}): {exc}") from exc

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", image_module.DecompressionBombWarning)
                image.load()
            orientation = int(image.getexif().get(274, 1) or 1)
            image_ops.exif_transpose(image, in_place=True)
        except (image_module.DecompressionBombWarning, image_module.DecompressionBombError) as exc:
            raise RuntimeError(
                f"Image rejected by Pillow decompression-bomb protection ({suffix or 'unknown suffix'})",
            ) from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to decode image ({suffix or 'unknown suffix'}): {exc}") from exc

        needs_convert = source_mime not in _PASSTHROUGH_IMAGE_MIMES
        needs_orientation = orientation in range(2, 9)
        if not needs_resize and not needs_convert and not needs_orientation:
            return None, source_mime, source_mime
        resize_frame = None
        try:
            frame = image
            if needs_resize:
                # Pillow forces NEAREST for palette and bilevel images, even
                # when LANCZOS is requested. Expand these modes within the
                # checked pixel budget so resizing retains fine strokes and
                # palette transparency. Other modes resize before conversion.
                if image.mode in ("P", "1"):
                    resize_frame = image.convert("RGBA" if image.mode == "P" else "L")
                    frame = resize_frame
                # Pillow 10 cannot apply LANCZOS directly to 16-bit integer
                # modes; NEAREST keeps that path bounded without a full-size
                # RGB conversion first.
                resize_filter = image_module.Resampling.LANCZOS
                if frame.mode.startswith("I;16"):
                    resize_filter = image_module.Resampling.NEAREST
                frame.thumbnail((MAX_IMAGE_REQUEST_DIMENSION, MAX_IMAGE_REQUEST_DIMENSION), resize_filter)
            has_alpha = frame.mode in ("RGBA", "LA", "P")
            frame = frame.convert("RGBA" if has_alpha else "RGB")
            try:
                buffer = io.BytesIO()
                if frame.mode == "RGBA":
                    frame.save(buffer, format="PNG")
                    return buffer.getvalue(), "image/png", source_mime
                frame.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
                return buffer.getvalue(), "image/jpeg", source_mime
            finally:
                frame.close()
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Failed to convert/resize image ({suffix or 'unknown suffix'}): {exc}") from exc
        finally:
            if resize_frame is not None:
                resize_frame.close()


def _build_image_request_payload(
    data: bytes,
    suffix: str,
    *,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
) -> dict:
    """Return ``{"data_b64", "mime", "source_mime", "converted"}`` for a VLM request.

    ``mime`` is the format actually sent (after in-memory downscale/re-encode);
    ``source_mime`` is the decoded format of the stored resource file and is
    what notes record. Both are based on actual bytes, not the filename suffix.
    """
    normalized_bytes, mime, source_mime = _normalize_image_bytes(
        data,
        suffix,
        max_image_pixels=max_image_pixels,
    )
    request_bytes = data if normalized_bytes is None else normalized_bytes
    return {
        "data_b64": base64.b64encode(request_bytes).decode("ascii"),
        "mime": mime,
        "source_mime": source_mime,
        "converted": normalized_bytes is not None,
    }


@R.register("auto_image_resource_step")
class AutoImageResourceStep(BaseAutoResourceStep):
    """Prepare native image inputs for the shared note-writing agent."""

    resource_suffixes = IMAGE_SUFFIXES
    router_inherit_keys = BaseAutoResourceStep.router_inherit_keys | frozenset(
        {"agent_wrapper", "max_image_bytes", "max_image_pixels", "prompt_dict"},
    )

    def _max_image_bytes(self) -> int:
        """Return the image read limit from Step or Job context."""
        value = self.kwargs.get("max_image_bytes")
        if value is None and self.context is not None:
            value = self.context.get("max_image_bytes")
        return int(value) if value is not None else DEFAULT_MAX_IMAGE_INPUT_BYTES

    def _max_image_pixels(self) -> int:
        """Return the deployment-controlled pre-decode pixel limit."""
        value = self.kwargs.get("max_image_pixels", DEFAULT_MAX_IMAGE_PIXELS)
        try:
            limit = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_image_pixels must be a positive integer: {value!r}") from exc
        if limit <= 0:
            raise ValueError(f"max_image_pixels must be a positive integer: {value!r}")
        return limit

    def _skip_resource_change(self, file_path: str) -> bool:
        """Disabling image inputs skips the whole image lifecycle, including deletes."""
        if self.context.get("include_images", True) is not False:
            return False
        self.context.response.success = True
        self.context.response.answer = f"Skipped image resource: {file_path} (include_images=false)"
        self.context.response.metadata.update(
            {"path": file_path, "action": "skipped", "reason": "include_images=false", "modified": False},
        )
        self.logger.warning(f"[{self.name}] skipped image file_path={file_path} reason=include_images=false")
        return True

    async def _read_image(self, file_path: str, source_path: Path) -> dict | None:
        """Read the image file and build the VLM request payload.

        Returns ``None`` when the change must be skipped (stat failure or
        oversized file); the skip outcome is already recorded on the response.
        """
        max_image_bytes = self._max_image_bytes()
        try:
            size_bytes = source_path.stat().st_size
        except OSError as exc:
            self.context.response.success = False
            self.context.response.answer = f"Failed to inspect resource file: {file_path}: {exc}"
            self.context.response.metadata.update(
                {
                    "path": file_path,
                    "action": "failed",
                    "error": str(exc),
                    "modified": False,
                },
            )
            self.logger.warning(f"[{self.name}] resource stat failed file_path={file_path} error={exc}")
            return None
        if size_bytes > max_image_bytes:
            self._record_oversized_image(file_path, size_bytes, max_image_bytes)
            return None

        self.logger.info(f"[{self.name}] read image start file_path={file_path}")
        async with aiofiles.open(source_path, "rb") as f:
            data = await f.read(max_image_bytes + 1)
        if len(data) > max_image_bytes:
            self._record_oversized_image(file_path, len(data), max_image_bytes)
            return None
        payload = _build_image_request_payload(
            data,
            Path(file_path).suffix.lower(),
            max_image_pixels=self._max_image_pixels(),
        )
        self.logger.info(
            f"[{self.name}] read image done file_path={file_path} size_bytes={size_bytes} "
            f"mime={payload['mime']} source_mime={payload['source_mime']} converted={payload['converted']}",
        )
        return payload

    def _record_oversized_image(self, file_path: str, size_bytes: int, max_image_bytes: int) -> None:
        """Record a stable skip response for an image over the compressed-byte limit."""
        assert self.context is not None
        self.context.response.success = True
        self.context.response.answer = (
            f"Skipped oversized image resource file: {file_path} ({size_bytes} > {max_image_bytes} bytes)"
        )
        self.context.response.metadata.update(
            {
                "path": file_path,
                "action": "skipped",
                "reason": "file_too_large",
                "oversized": True,
                "size_bytes": size_bytes,
                "max_image_bytes": max_image_bytes,
                "modified": False,
            },
        )
        self.logger.warning(
            f"[{self.name}] skip oversized image resource file_path={file_path} "
            f"size_bytes={size_bytes} max_image_bytes={max_image_bytes}",
        )

    async def _handle_upsert(
        self,
        file_path: str,
        date_str: str,
        note_stem: str,
        added: bool,
        source_path: Path,
    ) -> None:
        """Prepare a bounded image message, then use the common note-writing agent."""
        wrapper = self.agent_wrapper
        if wrapper is None or wrapper.backend != "agentscope":
            raise NotImplementedError("Image resources require the AgentScope wrapper")
        config = ContextConfig(**(wrapper.kwargs.get("context_config") or {}))
        if config.max_image_num < 1:
            raise ValueError("Image resource exceeds context_config.max_image_num; configure the wrapper explicitly")
        payload = await self._read_image(file_path, source_path)
        if payload is None:
            return
        blocks = [
            DataBlock(
                source=Base64Source(data=payload["data_b64"], media_type=payload["mime"]),
                name="image",
            ),
        ]
        note_path = await self._interpret_resource(
            file_path,
            date_str,
            note_stem,
            added,
            "The resource is the image attached above.",
            input_blocks=blocks,
            resource_instructions=self.prompt_format(
                "resource_instructions",
                file_path=file_path,
                filename=PurePosixPath(file_path).name,
                stem=note_stem,
                date=date_str,
            ),
            note_metadata={"kind": "image", "media_type": payload["source_mime"]},
            reply_kwargs={
                "scope_note_tools": True,
                "session_id": None,
                "resume": None,
                "builtin_tools": [],
                "skills": [],
                "toolkit": None,
                "output_schema": None,
            },
        )
        if note_path is None:
            raise RuntimeError("Resource agent did not write a note")
        self.context.response.metadata["media_type"] = payload["source_mime"]

    def _validate_resource_note(self, path: str, file_path: str, before_bytes: bytes | None) -> None:
        """Accept the written caption, without rewriting it or changing downstream status."""
        post = frontmatter.loads((self._note_bytes(path) or b"").decode("utf-8"))
        lines = [line.strip() for line in post.content.splitlines() if line.strip()]
        if lines[:2] != [f"![[{file_path}]]", "## Caption"]:
            raise ValueError("Image note must begin with the source image embed followed by '## Caption'")
        caption = "\n".join(lines[2:])
        if not caption:
            raise ValueError("Image note caption must not be empty")
        # A complete JSON payload is not a caption; prose containing OCR code blocks is valid.
        if len(lines) >= 4 and lines[2].lower() in {"```", "```json", "~~~", "~~~json"}:
            if lines[-1] == lines[2][:3]:
                caption = "\n".join(lines[3:-1])
                if not caption:
                    raise ValueError("Image note caption must not be empty")
        try:
            payload = json.loads(caption)
        except ValueError:
            payload = None
        if isinstance(payload, (dict, list)):
            raise ValueError("Image note caption must be a description or transcription, not a JSON payload")
        before = frontmatter.loads((before_bytes or b"").decode("utf-8"))
        if ("status" in before) != ("status" in post) or before.get("status") != post.get("status"):
            raise ValueError("Image agent must preserve existing 'status' and must not add, change, or remove it")
