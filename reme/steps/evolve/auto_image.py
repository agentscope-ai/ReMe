"""auto_image — interpret image resource files into source-linked daily notes via a VLM."""

import base64
import io
import json
import re
from pathlib import Path

import aiofiles
from agentscope.message import Base64Source, DataBlock, TextBlock, UserMsg
from agentscope.model import ChatModelBase
from PIL import Image
from pydantic import BaseModel, Field

from ..file_io import is_image_file, refresh_day_index
from ..file_io._path import IMAGE_MIME_BY_EXT
from .auto_resource import _SOURCE_RESOURCE_KEY, _sanitize_note_name, AutoResourceStep
from ...components import R
from ...enumeration import ComponentEnum

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    # Without the plugin HEIC files stay passthrough and the provider rejects
    # them per change; the pipeline itself keeps importing and running.
    pass

DEFAULT_MAX_IMAGE_INPUT_BYTES = 50 * 1024 * 1024
MAX_IMAGE_REQUEST_DIMENSION = 2048
_JPEG_QUALITY = 85
# Suffixes re-encoded to PNG for VLM requests; the stored resource file is never modified.
_CONVERT_SUFFIXES = {".bmp", ".tiff", ".heic"}
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class _CaptionOutput(BaseModel):
    """Structured caption contract enforced on the vision model."""

    name: str = Field(description="short kebab-case topic stem for the note filename; never include dates")
    description: str = Field(description="one-sentence summary that conveys the key information on its own")
    caption: str = Field(description="complete description / verbatim transcription of meaningful visible text")


def _normalize_image_bytes(data: bytes, suffix: str) -> tuple[bytes, str] | None:
    """Downscale or re-encode image bytes in memory for a VLM request.

    Returns ``None`` to pass the original bytes through untouched (already a
    reasonable size and format, or content that PIL cannot decode — left for
    the provider to reject). The stored resource file is never modified.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            needs_resize = image.width > MAX_IMAGE_REQUEST_DIMENSION or image.height > MAX_IMAGE_REQUEST_DIMENSION
            needs_convert = suffix in _CONVERT_SUFFIXES
            if not needs_resize and not needs_convert:
                return None
            has_alpha = image.mode in ("RGBA", "LA", "P")
            frame = image.convert("RGBA" if has_alpha else "RGB")
            if needs_resize:
                frame.thumbnail((MAX_IMAGE_REQUEST_DIMENSION, MAX_IMAGE_REQUEST_DIMENSION), Image.LANCZOS)
            buffer = io.BytesIO()
            if frame.mode == "RGBA":
                frame.save(buffer, format="PNG")
                return buffer.getvalue(), "image/png"
            frame.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
            return buffer.getvalue(), "image/jpeg"
    except Exception:  # pylint: disable=broad-except
        return None


def _build_image_request_payload(data: bytes, suffix: str) -> dict:
    """Return ``{"data_b64", "mime", "source_mime", "converted"}`` for a VLM request.

    ``mime`` is the format actually sent (after in-memory downscale/re-encode);
    ``source_mime`` describes the stored resource file and is what notes record.
    """
    source_mime = IMAGE_MIME_BY_EXT.get(suffix, "image/png")
    normalized = _normalize_image_bytes(data, suffix)
    if normalized is None:
        return {
            "data_b64": base64.b64encode(data).decode("ascii"),
            "mime": source_mime,
            "source_mime": source_mime,
            "converted": False,
        }
    normalized_bytes, mime = normalized
    return {
        "data_b64": base64.b64encode(normalized_bytes).decode("ascii"),
        "mime": mime,
        "source_mime": source_mime,
        "converted": normalized_bytes != data,
    }


async def _response_text(result) -> str:
    """Extract text blocks from a streaming or non-streaming ChatResponse."""
    if hasattr(type(result), "__aiter__"):
        last = None
        async for chunk in result:
            last = chunk
        result = last
    if result is None:
        return ""
    parts: list[str] = []
    for block in result.content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        elif getattr(block, "type", None) == "text":
            parts.append(str(getattr(block, "text", "") or ""))
    return "".join(parts).strip()


def _normalize_caption_fields(parsed: dict) -> dict:
    """Normalize parsed caption fields, cross-filling a missing ``caption``
    from a present ``description`` so raw JSON never reaches the note body."""
    caption = str(parsed.get("caption") or "").strip()
    description = str(parsed.get("description") or "").strip()
    if not caption and description:
        caption = description
    return {
        "name": str(parsed.get("name") or "").strip(),
        "description": description,
        "caption": caption,
    }


def _parse_caption_json(text: str) -> dict:
    """Parse a plain-call caption response leniently.

    Used as the fallback when the schema-forced structured call fails: fenced
    JSON and embedded ``{...}`` slices are tried before degrading the whole
    response text to the caption.
    """
    cleaned = text.strip()
    fence = _JSON_FENCE_RE.match(cleaned)
    if fence:
        cleaned = fence.group(1)
    for candidate in (cleaned, cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            normalized = _normalize_caption_fields(parsed)
            if normalized["caption"] or normalized["description"]:
                return normalized
    return {"name": "", "description": "", "caption": text.strip()}


@R.register("auto_image_step")
class AutoImageStep(AutoResourceStep):
    """Interpret image resource files into daily notes via a direct VLM call.

    Unlike text resources (agent + file tools), the image interpretation is a
    single vision-model call. Images larger than the request budget or in
    provider-unfriendly formats are downscaled/re-encoded in memory for the
    request only; files under ``resource/`` are never modified. Note lookup,
    renaming, deletion linkage, and day-index refresh reuse the inherited
    AutoResourceStep helpers; only the interpretation differs.
    """

    def _skip_image_change(self, file_path: str) -> bool:
        """This step interprets images, so the inherited image guard never applies."""
        return False

    def _max_image_bytes(self) -> int:
        """Return the image read limit from Step or Job context."""
        value = self.kwargs.get("max_image_bytes")
        if value is None and self.context is not None:
            value = self.context.get("max_image_bytes")
        return int(value) if value is not None else DEFAULT_MAX_IMAGE_INPUT_BYTES

    def _vision_model(self) -> ChatModelBase | None:
        """Resolve the vision model: an explicit instance, then the ``vision``
        named as_llm component, then the ``default`` one."""
        for source in (self.kwargs, self.context or {}):
            value = source.get("as_llm")
            if isinstance(value, ChatModelBase):
                return value
        if self.app_context is None:
            return None
        models = self.app_context.components.get(ComponentEnum.AS_LLM, {})
        component = models.get("vision") or models.get("default")
        if component is None:
            return None
        return getattr(component, "model", None)

    async def _caption_with_retry(self, model: ChatModelBase, user_message: UserMsg) -> dict:
        """Return the caption fields from the vision model.

        Primary path is the schema-forced structured output (the SDK enforces
        the ``name``/``description``/``caption`` contract and retries transport
        errors). When that fails or yields no usable field, retry once with a
        plain call parsed leniently.
        """
        try:
            structured = await model.generate_structured_output(
                messages=[user_message],
                structured_model=_CaptionOutput,
            )
            content = structured.content if isinstance(structured.content, dict) else {}
            normalized = _normalize_caption_fields(dict(content))
            if normalized["caption"] or normalized["description"]:
                self.logger.info(f"[{self.name}] structured caption ok name={normalized['name']}")
                return normalized
            self.logger.warning(f"[{self.name}] structured caption empty; retrying with a plain call")
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(f"[{self.name}] structured caption failed ({exc}); retrying with a plain call")
        result = await model([user_message])
        return _parse_caption_json(await _response_text(result))

    async def _read_image(self, file_path: str) -> dict | None:
        """Read the image file and build the VLM request payload.

        Returns ``None`` when the change must be skipped (stat failure or
        oversized file); the skip outcome is already recorded on the response.
        """
        abs_path = self.workspace_path / file_path
        max_image_bytes = self._max_image_bytes()
        try:
            size_bytes = abs_path.stat().st_size
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
            return None

        self.logger.info(f"[{self.name}] read image start file_path={file_path}")
        async with aiofiles.open(abs_path, "rb") as f:
            data = await f.read()
        payload = _build_image_request_payload(data, Path(file_path).suffix.lower())
        self.logger.info(
            f"[{self.name}] read image done file_path={file_path} size_bytes={size_bytes} "
            f"mime={payload['mime']} converted={payload['converted']}",
        )
        return payload

    async def _handle_change(self, file_path: str, raw_change) -> dict:
        """Skip non-image changes; isolate per-change failures so the batch continues."""
        if file_path and not is_image_file(file_path):
            file_path = self.to_workspace_relative(file_path) if Path(file_path).is_absolute() else file_path
            self.context.response.metadata = {}
            answer = f"Skipped non-image resource file: {file_path}"
            self.context.response.success = True
            self.context.response.answer = answer
            self.context.response.metadata.update(
                {
                    "path": file_path,
                    "action": "skipped",
                    "reason": "non_image_file",
                    "modified": False,
                },
            )
            self.logger.info(f"[{self.name}] skip change file_path={file_path} reason=non_image_file")
            return {
                "success": True,
                "path": file_path,
                "change": str(raw_change),
                "answer": answer,
                "metadata": dict(self.context.response.metadata),
            }
        try:
            return await super()._handle_change(file_path, raw_change)
        except Exception as exc:  # pylint: disable=broad-except
            self.context.response.success = False
            self.context.response.answer = f"Failed to caption image resource: {file_path}: {exc}"
            self.context.response.metadata.update(
                {
                    "path": file_path,
                    "action": "failed",
                    "error": str(exc),
                    "modified": False,
                },
            )
            self.logger.warning(f"[{self.name}] caption failed file_path={file_path} error={exc}")
            return {
                "success": False,
                "path": file_path,
                "change": str(raw_change),
                "answer": self.context.response.answer,
                "metadata": dict(self.context.response.metadata),
            }

    async def _handle_upsert(self, file_path: str, date_str: str, note_stem: str, added: bool) -> None:
        """Caption the image and write/refresh its note (image counterpart of the text upsert)."""
        daily_dir = self.config_value("daily_dir")
        fallback_path = f"{daily_dir}/{date_str}/{note_stem}.md"
        try:
            note = await self._list_resource_note(date_str, file_path, fallback_path)
        except RuntimeError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            self.logger.info(f"[{self.name}] list failed file_path={file_path} answer={str(exc)!r}")
            return

        note_path = str(note["path"]) if note else fallback_path
        note_created = note is None
        before_note_path = note_path
        before_note_bytes = self._note_bytes(note_path)
        self.logger.info(
            f"[{self.name}] upsert start file_path={file_path} date={date_str} " f"note_stem={note_stem} added={added}",
        )

        model = self._vision_model()
        if model is None:
            self.context.response.success = True
            self.context.response.answer = f"Skipped image resource without a vision model: {file_path}"
            self.context.response.metadata.update(
                {
                    "path": file_path,
                    "action": "skipped",
                    "reason": "vision_model_not_configured",
                    "modified": False,
                },
            )
            self.logger.warning(f"[{self.name}] no vision model configured file_path={file_path}")
            return

        payload = await self._read_image(file_path)
        if payload is None:
            return

        user_message = UserMsg(
            name="user",
            content=[
                TextBlock(text=self.prompt_format("user_message", file_path=file_path, date=date_str)),
                DataBlock(
                    source=Base64Source(data=payload["data_b64"], media_type=payload["mime"]),
                    name="image",
                ),
            ],
        )
        parsed = await self._caption_with_retry(model, user_message)
        name = _sanitize_note_name(str(parsed.get("name") or ""), note_stem)
        caption = str(parsed.get("caption") or "").strip()
        description = str(parsed.get("description") or "").strip() or caption[:120]
        body = f"![[{file_path}]]\n\n## Caption\n\n{caption}\n"

        # The write job's ``name`` parameter is the note name; calling the job
        # directly (instead of run_job) keeps it clear of run_job's
        # positional-only job-selector argument.
        write_job = self.get_job("write")
        if write_job is None:
            raise RuntimeError("Job write not found")
        write_response = await write_job(
            path=note_path,
            name=name,
            description=description,
            content=body,
            metadata={
                _SOURCE_RESOURCE_KEY: self._source_resource_link(file_path),
                "kind": "image",
                "media_type": payload["source_mime"],
            },
        )
        if not write_response.success:
            raise RuntimeError(f"write failed: {write_response.answer}")

        if note_created:
            try:
                note = await self._list_resource_note(date_str, file_path, fallback_path)
            except RuntimeError as exc:
                self.context.response.success = False
                self.context.response.answer = str(exc)
                self.context.response.metadata.update({"path": None, "created": note_created, "modified": False})
                self.logger.info(f"[{self.name}] post-create list failed file_path={file_path} answer={str(exc)!r}")
                return
            if note is None:
                self.context.response.success = True
                self.context.response.answer = f"Captioned image resource {file_path}"
                self.context.response.metadata.update({"path": None, "created": False, "modified": False})
                self.logger.info(f"[{self.name}] done without note file_path={file_path} modified=False")
                return
            note_path = str(note["path"])

        try:
            await self._ensure_resource_frontmatter(note_path, file_path)
            note_path = await self._rename_from_frontmatter_name(
                note_path,
                date_str,
                file_path,
                note_stem,
                fallback_path,
                allow_rename=note_created,
            )
        except RuntimeError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            self.context.response.metadata.update(
                {
                    "path": note_path,
                    "created": note_created,
                    "modified": self._note_modified(before_note_path, before_note_bytes, note_path),
                },
            )
            self.logger.info(f"[{self.name}] post-write failed path={note_path} answer={str(exc)!r}")
            return

        modified = self._note_modified(before_note_path, before_note_bytes, note_path)
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)

        self.context.response.success = True
        self.context.response.answer = f"Captioned image resource {file_path} -> {note_path}"
        self.context.response.metadata.update(
            {
                "path": note_path,
                "created": note_created,
                "modified": modified,
                "session_id": note_stem,
                "source_resource": self._source_resource_link(file_path),
                "action": "added" if added else "modified",
                "media_type": payload["source_mime"],
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path} modified={modified}")
