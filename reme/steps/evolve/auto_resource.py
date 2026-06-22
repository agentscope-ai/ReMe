"""auto_resource — interpret resource files into same-name daily notes via an agent."""

import uuid
from pathlib import Path, PurePosixPath

import aiofiles
from watchfiles import Change

from ..base_step import BaseStep
from ..file_io import refresh_day_index
from ...components import R


def _compute_agent_session_id(path: str) -> str:
    """Return a stable UUID session id for agent backends."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, path))


def _compute_note_stem(filename: str) -> str:
    """Return the daily note stem for a resource filename."""
    return PurePosixPath(filename).stem


def _parse_resource_path(file_path: str, resource_dir: str) -> tuple[str, str]:
    """Extract (date, filename) from a resource path like 'resource/2026-06-06/report.pdf'.

    Returns (date_str, filename) where filename may contain subdirectories.
    """
    parts = PurePosixPath(file_path).parts
    # Strip leading resource_dir prefix
    prefix_parts = PurePosixPath(resource_dir).parts
    if parts[: len(prefix_parts)] == prefix_parts:
        parts = parts[len(prefix_parts) :]
    # First segment is date, rest is filename
    date_str = parts[0] if parts else ""
    filename = str(PurePosixPath(*parts[1:])) if len(parts) > 1 else ""
    return date_str, filename


@R.register("auto_resource_step")
class AutoResourceStep(BaseStep):
    """Interpret resource files into daily notes via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    def _normalize_change(self, raw) -> Change | None:
        if isinstance(raw, Change):
            return raw
        if isinstance(raw, str):
            return Change.__members__.get(raw)
        return None

    async def _handle_delete(self, date_str: str, note_stem: str) -> None:
        daily_dir = self.config_value("daily_dir")
        note_rel = f"{daily_dir}/{date_str}/{note_stem}.md"
        note_abs = self.workspace_path / note_rel

        if note_abs.is_file():
            note_abs.unlink()
            self.logger.info(f"[{self.name}] Deleted file: {note_rel}")

        await self.file_store.delete([note_rel])
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)

        self.context.response.success = True
        self.context.response.answer = f"Deleted resource note: {note_rel}"
        self.context.response.metadata.update(
            {"path": note_rel, "session_id": note_stem, "action": "deleted", "index": index_payload},
        )

    async def _handle_upsert(self, file_path: str, date_str: str, note_stem: str, created: bool) -> None:
        create_response = await self.run_job("daily_create", session_id=note_stem, date=date_str)
        if not create_response.success:
            self.context.response.success = False
            self.context.response.answer = f"daily_create failed: {create_response.answer}"
            return

        note_path: str = create_response.metadata["path"]
        note_created: bool = create_response.metadata["created"]

        # Read resource file content
        abs_path = self.workspace_path / file_path
        if not abs_path.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Resource file not found: {file_path}"
            return

        async with aiofiles.open(abs_path, encoding="utf-8", errors="replace") as f:
            file_content = await f.read()

        template_key = "user_message_create" if created else "user_message_update"
        user_message = self.prompt_format(
            template_key,
            workspace_dir=str(self.workspace_path),
            note_path=note_path,
            file_path=file_path,
            file_content=file_content,
            date=date_str,
        )

        agent_session_id = _compute_agent_session_id(file_path)
        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            job_tools=self.agent_tools,
            session_id=agent_session_id,
        )
        daily_dir = self.config_value("daily_dir")
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)

        self.context.response.success = True
        self.context.response.answer = (result.get("result") or "").strip()
        self.context.response.metadata.update(
            {
                "path": note_path,
                "created": note_created,
                "session_id": note_stem,
                "agent_session_id": agent_session_id,
                "action": "added" if created else "modified",
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path}")

    async def _handle_change(self, file_path: str, raw_change) -> dict:
        assert self.context is not None
        file_path = self.to_workspace_relative(file_path) if file_path and Path(file_path).is_absolute() else file_path
        if not file_path:
            self.context.response.success = False
            self.context.response.answer = "Missing file_path"
            return {"success": False, "path": file_path, "change": raw_change, "answer": self.context.response.answer}

        change = self._normalize_change(raw_change)
        if change is None:
            self.context.response.success = False
            self.context.response.answer = f"Invalid change type: {raw_change}"
            return {"success": False, "path": file_path, "change": raw_change, "answer": self.context.response.answer}

        resource_dir = self.config_value("resource_dir")
        date_str, filename = _parse_resource_path(file_path, resource_dir)

        if not date_str or not filename:
            self.context.response.success = False
            self.context.response.answer = f"Cannot parse date/filename from: {file_path}"
            return {"success": False, "path": file_path, "change": change.name, "answer": self.context.response.answer}

        note_stem = _compute_note_stem(filename)
        self.logger.info(f"[{self.name}] {change.name} file_path={file_path} note_stem={note_stem}")

        if change == Change.deleted:
            await self._handle_delete(date_str, note_stem)
        else:
            await self._handle_upsert(
                file_path,
                date_str,
                note_stem,
                created=change == Change.added,
            )
        return {
            "success": self.context.response.success,
            "path": file_path,
            "change": change.name,
            "answer": self.context.response.answer,
            "metadata": dict(self.context.response.metadata),
        }

    async def execute(self):
        assert self.context is not None
        changes = self.context.get("changes")
        if not isinstance(changes, list):
            self.context.response.success = False
            self.context.response.answer = "AutoResourceStep requires changes: list[dict]"
            return self.context.response

        results = [
            await self._handle_change(item.get("path") or item.get("file_path", ""), item.get("change", ""))
            for item in changes
            if isinstance(item, dict)
        ]
        success_count = sum(1 for item in results if item.get("success"))
        self.context.response.success = success_count == len(changes)
        self.context.response.answer = f"Processed {success_count}/{len(changes)} resource change(s)"
        self.context.response.metadata["processed"] = len(results)
        self.context.response.metadata["results"] = results
        return self.context.response
