"""Text resource processor for the unified auto-resource router."""

import uuid
from pathlib import Path

import aiofiles

from ...components import R
from ._evolve import agent_reply_result_text
from .base_auto_resource import BaseAutoResourceStep


def _compute_agent_session_id(path: str) -> str:
    """Return a stable UUID session id for agent backends."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


@R.register("auto_text_resource_step")
class AutoTextResourceStep(BaseAutoResourceStep):
    """Interpret text resource files into daily notes via an Agent."""

    # Preserve the pre-router AutoResourceStep behavior for direct calls and
    # custom watcher suffixes; the default watcher still limits normal inputs.
    resource_fallback = True
    router_inherit_keys = BaseAutoResourceStep.router_inherit_keys | frozenset(
        {"agent_wrapper", "max_file_bytes", "prompt_dict"},
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_tools: list[str] = ["write"]
        self.update_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

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
        note_state = await self._prepare_resource_note(date_str, file_path, note_stem)
        note_path = note_state.path
        note_created = note_state.created
        self.logger.info(f"[{self.name}] daily note lookup path={note_path} created={note_created}")

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

        template_key = "user_message_create" if note_created else "user_message_update"
        user_message = self.prompt_format(
            template_key,
            workspace_dir=str(self.workspace_path),
            note_path=note_path,
            note_stem=note_stem,
            file_path=file_path,
            source_resource=self._source_resource_link(file_path),
            file_content=file_content,
            date=date_str,
        )

        agent_session_id = _compute_agent_session_id(file_path)
        self.logger.info(
            f"[{self.name}] agent start file_path={file_path} note_path={note_path} "
            f"agent_session_id={agent_session_id}",
        )
        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            job_tools=self.create_tools if note_created else self.update_tools,
            session_id=agent_session_id,
        )
        self.logger.info(f"[{self.name}] agent done file_path={file_path} has_result={bool(result.get('result'))}")

        note_path = await self._finalize_resource_note(
            note_state,
            date_str,
            file_path,
            note_stem,
            added,
        )
        if note_path is None:
            self.context.response.success = True
            self.context.response.answer = agent_reply_result_text(result)
            self.logger.info(f"[{self.name}] done without note file_path={file_path} modified=False")
            return

        self.context.response.success = True
        self.context.response.answer = agent_reply_result_text(result)
        self.context.response.metadata.update(
            {
                "agent_session_id": agent_session_id,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path} modified={self.context.response.metadata['modified']}")
