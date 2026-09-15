"""Text resource processor for the unified auto-resource router."""

from pathlib import Path

import aiofiles

from ...components import R
from .base_auto_resource import BaseAutoResourceStep


@R.register("auto_text_resource_step")
class AutoTextResourceStep(BaseAutoResourceStep):
    """Interpret text resource files into daily notes via an Agent."""

    # Preserve the pre-router AutoResourceStep behavior for direct calls and
    # custom watcher suffixes; the default watcher still limits normal inputs.
    resource_fallback = True
    router_inherit_keys = BaseAutoResourceStep.router_inherit_keys | frozenset(
        {"agent_wrapper", "max_file_bytes", "prompt_dict"},
    )

    async def _handle_upsert(
        self,
        file_path: str,
        date_str: str,
        note_stem: str,
        added: bool,
        source_path: Path,
    ) -> None:
        self.logger.info(
            f"[{self.name}] upsert start file_path={file_path} date={date_str} " f"note_stem={note_stem} added={added}",
        )
        # Read resource file content
        if not source_path.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Resource file not found: {file_path}"
            self.logger.warning(f"[{self.name}] resource missing file_path={file_path}")
            return

        skip_read = False
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
            skip_read = True
        if not skip_read:
            max_file_bytes = self.max_file_bytes()
            if size_bytes > max_file_bytes:
                self.context.response.success = True
                self.context.response.answer = (
                    f"Skipped oversized resource file: {file_path} ({size_bytes} > {max_file_bytes} bytes)"
                )
                self.context.response.metadata.update(
                    {
                        "path": file_path,
                        "action": "skipped",
                        "reason": "file_too_large",
                        "oversized": True,
                        "size_bytes": size_bytes,
                        "max_file_bytes": max_file_bytes,
                        "modified": False,
                    },
                )
                self.logger.warning(
                    f"[{self.name}] skip oversized resource file_path={file_path} "
                    f"size_bytes={size_bytes} max_file_bytes={max_file_bytes}",
                )
                skip_read = True
        if skip_read:
            return

        self.logger.info(f"[{self.name}] read resource start file_path={file_path}")
        async with aiofiles.open(source_path, encoding="utf-8", errors="replace") as f:
            file_content = await f.read()
        self.logger.info(f"[{self.name}] read resource done file_path={file_path} chars={len(file_content)}")

        await self._interpret_resource(file_path, date_str, note_stem, added, file_content)
