"""Text resource processor for the unified auto-resource router."""

import uuid

import aiofiles

from ...components import R
from ._evolve import agent_reply_result_text
from ._auto_resource import BaseAutoResourceStep


def _compute_agent_session_id(path: str) -> str:
    """Return a stable UUID session id for agent backends."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


@R.register("auto_text_resource_step")
class AutoTextResourceStep(BaseAutoResourceStep):
    """Interpret text resource files into daily notes via an Agent."""

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
    ) -> None:
        self.logger.info(
            f"[{self.name}] upsert start file_path={file_path} date={date_str} " f"note_stem={note_stem} added={added}",
        )
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
        self.logger.info(f"[{self.name}] daily note lookup path={note_path} created={note_created}")

        # Read resource file content
        abs_path = self.workspace_path / file_path
        if not abs_path.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Resource file not found: {file_path}"
            self.logger.warning(f"[{self.name}] resource missing file_path={file_path}")
            return

        skip_read = False
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
        async with aiofiles.open(abs_path, encoding="utf-8", errors="replace") as f:
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
                self.context.response.answer = agent_reply_result_text(result)
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
            self.logger.info(f"[{self.name}] post-agent failed path={note_path} answer={str(exc)!r}")
            return

        modified = self._note_modified(before_note_path, before_note_bytes, note_path)
        index_payload = await self._refresh_day_index(date_str)

        self.context.response.success = True
        self.context.response.answer = agent_reply_result_text(result)
        self.context.response.metadata.update(
            {
                "path": note_path,
                "created": note_created,
                "modified": modified,
                "session_id": note_stem,
                "source_resource": self._source_resource_link(file_path),
                "agent_session_id": agent_session_id,
                "action": "added" if added else "modified",
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path} modified={modified}")
