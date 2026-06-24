"""auto_resource — interpret resource files into same-name daily notes via an agent."""

import inspect
import uuid
from pathlib import Path, PurePosixPath

import aiofiles
from watchfiles import Change

from ..base_step import BaseStep
from ..file_io import refresh_day_index
from ...components import R
from ._evolve import agent_reply_result_text, now


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


def _loose_resource_filename(file_path: str, resource_dir: str) -> str:
    """Return filename for a root-level resource path like 'resource/report.txt'."""
    parts = PurePosixPath(file_path).parts
    prefix_parts = PurePosixPath(resource_dir).parts
    if parts[: len(prefix_parts)] != prefix_parts:
        return ""
    rest = parts[len(prefix_parts) :]
    if len(rest) != 1:
        return ""
    filename = rest[0]
    return "" if filename in ("", ".", "..") else filename


def _results_answer(results: list[dict], processed_answer: str) -> str:
    """Return the actual per-change answer while preserving a batch fallback."""
    answers = [str(item.get("answer") or "").strip() for item in results]
    answers = [item for item in answers if item]
    if len(answers) == 1:
        return answers[0]
    if len(answers) > 1:
        return "\n\n".join(f"{index}. {answer}" for index, answer in enumerate(answers, start=1))
    return processed_answer


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

    def _today(self) -> str:
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        return now(tz).strftime("%Y-%m-%d")

    async def _emit_result_hook(self, *, changes: list[dict], results: list[dict]) -> None:
        """Notify embedding hosts about the final auto-resource response.

        The hook is intentionally optional so standalone ReMe and old configs
        keep the existing behavior.
        """
        if self.app_context is None or self.context is None:
            return
        metadata = getattr(self.app_context, "metadata", None)
        if not isinstance(metadata, dict):
            return
        hook = metadata.get("qwenpaw_memory_result_hook")
        if hook is None:
            return
        try:
            value = hook(
                job_name="auto_resource",
                response=self.context.response,
                kwargs={"changes": changes},
                metadata={"results": results},
            )
            if inspect.isawaitable(value):
                await value
        except Exception:
            self.logger.exception(f"[{self.name}] result hook failed")

    async def _handle_delete(self, date_str: str, note_stem: str) -> None:
        daily_dir = self.config_value("daily_dir")
        note_rel = f"{daily_dir}/{date_str}/{note_stem}.md"
        note_abs = self.workspace_path / note_rel
        self.logger.info(f"[{self.name}] delete start note={note_rel}")

        if note_abs.is_file():
            note_abs.unlink()
            self.logger.info(f"[{self.name}] Deleted file: {note_rel}")

        await self.file_store.delete([note_rel])
        self.logger.info(f"[{self.name}] catalog delete done note={note_rel}")
        self.logger.info(f"[{self.name}] refresh index start date={date_str} daily_dir={daily_dir}")
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)
        self.logger.info(f"[{self.name}] refresh index done date={date_str}")

        self.context.response.success = True
        self.context.response.answer = f"Deleted resource note: {note_rel}"
        self.context.response.metadata.update(
            {"path": note_rel, "session_id": note_stem, "action": "deleted", "index": index_payload},
        )

    async def _handle_upsert(self, file_path: str, date_str: str, note_stem: str, created: bool) -> None:
        self.logger.info(
            f"[{self.name}] upsert start file_path={file_path} date={date_str} "
            f"note_stem={note_stem} created={created}",
        )
        create_response = await self.run_job("daily_create", session_id=note_stem, date=date_str)
        if not create_response.success:
            self.context.response.success = False
            self.context.response.answer = f"daily_create failed: {create_response.answer}"
            self.logger.info(
                f"[{self.name}] daily_create failed file_path={file_path} answer={create_response.answer!r}",
            )
            return

        note_path: str = create_response.metadata["path"]
        note_created: bool = create_response.metadata["created"]
        self.logger.info(f"[{self.name}] daily note ready path={note_path} created={note_created}")

        # Read resource file content
        abs_path = self.workspace_path / file_path
        if not abs_path.is_file():
            self.context.response.success = False
            self.context.response.answer = f"Resource file not found: {file_path}"
            self.logger.warning(f"[{self.name}] resource missing file_path={file_path}")
            return

        self.logger.info(f"[{self.name}] read resource start file_path={file_path}")
        async with aiofiles.open(abs_path, encoding="utf-8", errors="replace") as f:
            file_content = await f.read()
        self.logger.info(f"[{self.name}] read resource done file_path={file_path} chars={len(file_content)}")

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
        self.logger.info(
            f"[{self.name}] agent start file_path={file_path} note_path={note_path} "
            f"agent_session_id={agent_session_id}",
        )
        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            job_tools=self.agent_tools,
            session_id=agent_session_id,
        )
        self.logger.info(f"[{self.name}] agent done file_path={file_path} has_result={bool(result.get('result'))}")
        daily_dir = self.config_value("daily_dir")
        self.logger.info(f"[{self.name}] refresh index start date={date_str} daily_dir={daily_dir}")
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)
        self.logger.info(f"[{self.name}] refresh index done date={date_str}")

        self.context.response.success = True
        self.context.response.answer = agent_reply_result_text(result)
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
            self.logger.warning(f"[{self.name}] missing file_path change={raw_change!r}")
            return {"success": False, "path": file_path, "change": raw_change, "answer": self.context.response.answer}

        change = self._normalize_change(raw_change)
        if change is None:
            self.context.response.success = False
            self.context.response.answer = f"Invalid change type: {raw_change}"
            self.logger.warning(f"[{self.name}] invalid change file_path={file_path} change={raw_change!r}")
            return {"success": False, "path": file_path, "change": raw_change, "answer": self.context.response.answer}

        resource_dir = self.config_value("resource_dir")
        loose_filename = _loose_resource_filename(file_path, resource_dir)
        if loose_filename:
            date_str, filename = self._today(), loose_filename
            self.logger.info(f"[{self.name}] loose resource file_path={file_path} date={date_str}")
        else:
            date_str, filename = _parse_resource_path(file_path, resource_dir)

        if not date_str or not filename:
            self.context.response.success = False
            self.context.response.answer = f"Cannot parse date/filename from: {file_path}"
            self.logger.warning(f"[{self.name}] parse path failed file_path={file_path} resource_dir={resource_dir}")
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
            self.logger.warning(f"[{self.name}] invalid changes payload type={type(changes).__name__}")
            return self.context.response

        self.logger.info(f"[{self.name}] start changes={len(changes)}")
        results = []
        for index, item in enumerate(changes, start=1):
            if not isinstance(item, dict):
                self.logger.warning(f"[{self.name}] skip invalid change item index={index} type={type(item).__name__}")
                continue
            self.logger.info(f"[{self.name}] process change {index}/{len(changes)}")
            results.append(
                await self._handle_change(item.get("path") or item.get("file_path", ""), item.get("change", "")),
            )
        success_count = sum(1 for item in results if item.get("success"))
        self.context.response.success = success_count == len(changes)
        processed_answer = f"Processed {success_count}/{len(changes)} resource change(s)"
        self.context.response.answer = _results_answer(results, processed_answer)
        self.context.response.metadata["processed"] = len(results)
        self.context.response.metadata["results"] = results
        await self._emit_result_hook(changes=changes, results=results)
        self.logger.info(f"[{self.name}] done success={success_count}/{len(changes)} processed={len(results)}")
        return self.context.response
