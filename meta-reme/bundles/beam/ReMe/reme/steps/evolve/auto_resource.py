"""auto_resource — interpret resource files into source-linked daily notes via an agent."""

import hashlib
import inspect
import re
import uuid
from pathlib import Path, PurePosixPath

import aiofiles
import frontmatter
from watchfiles import Change

from ..base_step import BaseStep
from ..file_io import refresh_day_index, validate_filename_component
from ...components import R
from ._evolve import agent_reply_result_text, now

_SOURCE_RESOURCE_KEY = "source_resource"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


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
    if parts[: len(prefix_parts)] != prefix_parts:
        return "", ""
    parts = parts[len(prefix_parts) :]
    # First segment is date, rest is filename
    date_str = parts[0] if parts else ""
    if not _DATE_RE.match(date_str):
        return "", ""
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


def _source_suffix(file_path: str) -> str:
    """Return a short stable suffix for source-path collision handling."""
    return hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:8]


def _sanitize_note_name(raw: str, fallback: str) -> str:
    """Return a safe single filename component from an LLM-suggested name."""
    name = str(raw or "").strip()
    name = _UNSAFE_FILENAME_CHARS.sub("-", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        name = str(fallback or "").strip()
    name = _UNSAFE_FILENAME_CHARS.sub("-", name).strip(" .")
    if not name or validate_filename_component(name, kind="name"):
        name = f"resource-{_source_suffix(fallback or raw or 'note')}"
    if validate_filename_component(name, kind="name"):
        name = f"resource-{_source_suffix(name)}"
    return name


@R.register("auto_resource_step")
class AutoResourceStep(BaseStep):
    """Interpret resource files into daily notes via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_tools: list[str] = ["write"]
        self.update_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    def _normalize_change(self, raw) -> Change | None:
        if isinstance(raw, Change):
            return raw
        if isinstance(raw, str):
            return Change.__members__.get(raw)
        return None

    def _today(self) -> str:
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        return now(tz).strftime("%Y-%m-%d")

    def _daily_note_path(self, day: str, name: str) -> str:
        return f"{self.config_value('daily_dir')}/{day}/{name}.md"

    @staticmethod
    def _source_resource_link(file_path: str) -> str:
        return f"[[{file_path}]]"

    def _frontmatter(self, path: str) -> dict:
        post = frontmatter.loads((self.file_store.workspace_path / path).read_text(encoding="utf-8"))
        return dict(post.metadata or {})

    def _note_bytes(self, path: str) -> bytes | None:
        note_path = self.file_store.workspace_path / path
        if not note_path.is_file():
            return None
        return note_path.read_bytes()

    def _note_modified(self, before_path: str, before_bytes: bytes | None, after_path: str) -> bool:
        if not after_path:
            return False
        after_bytes = self._note_bytes(after_path)
        if after_bytes is None:
            return before_bytes is not None
        return after_path != before_path or before_bytes != after_bytes

    def _find_resource_note(self, notes: list[dict], file_path: str, fallback_path: str) -> dict | None:
        source = self._source_resource_link(file_path)
        for note in notes:
            if str(note.get(_SOURCE_RESOURCE_KEY, "")).strip() == source:
                return note
        for note in notes:
            if str(note.get("path", "")).strip() == fallback_path:
                return note
        return None

    async def _list_resource_note(self, day: str, file_path: str, fallback_path: str) -> dict | None:
        list_response = await self.run_job("daily_list", date=day)
        if not list_response.success:
            raise RuntimeError(f"daily_list failed: {list_response.answer}")
        notes = list_response.metadata.get("notes") or []
        return self._find_resource_note(notes, file_path, fallback_path)

    async def _ensure_resource_frontmatter(self, path: str, file_path: str) -> None:
        metadata = {_SOURCE_RESOURCE_KEY: self._source_resource_link(file_path)}
        current = self._frontmatter(path)
        if all(current.get(key) == value for key, value in metadata.items()):
            return
        response = await self.run_job(
            "frontmatter_update",
            path=path,
            metadata=metadata,
        )
        if not response.success:
            raise RuntimeError(f"frontmatter_update failed: {response.answer}")

    async def _set_frontmatter_name(self, path: str, name: str) -> None:
        if self._frontmatter(path).get("name") == name:
            return
        response = await self.run_job("frontmatter_update", path=path, metadata={"name": name})
        if not response.success:
            raise RuntimeError(f"frontmatter_update failed: {response.answer}")

    def _unique_daily_note_path(self, day: str, name: str, file_path: str, current_path: str) -> tuple[str, str]:
        """Return a collision-free (name, path), preserving current_path when possible."""
        target_path = self._daily_note_path(day, name)
        target_abs = self.file_store.workspace_path / target_path
        if target_path == current_path or not target_abs.exists():
            return name, target_path

        suffixed = f"{name}--{_source_suffix(file_path)}"
        target_path = self._daily_note_path(day, suffixed)
        target_abs = self.file_store.workspace_path / target_path
        if target_path == current_path or not target_abs.exists():
            return suffixed, target_path

        for index in range(2, 100):
            candidate = f"{suffixed}-{index}"
            target_path = self._daily_note_path(day, candidate)
            target_abs = self.file_store.workspace_path / target_path
            if target_path == current_path or not target_abs.exists():
                return candidate, target_path
        raise RuntimeError(f"cannot allocate unique note name for: {name!r}")

    async def _rename_from_frontmatter_name(
        self,
        path: str,
        day: str,
        file_path: str,
        fallback_name: str,
        fallback_path: str,
        *,
        allow_rename: bool,
    ) -> str:
        meta = self._frontmatter(path)
        current_name = PurePosixPath(path).stem
        suggested_name = str(meta.get("name", "")).strip()

        if not allow_rename and path != fallback_path:
            name = _sanitize_note_name(current_name, fallback_name)
            if suggested_name != name:
                await self._set_frontmatter_name(path, name)
            return path

        name = _sanitize_note_name(suggested_name, fallback_name)
        name, target_path = self._unique_daily_note_path(day, name, file_path, path)
        if suggested_name != name:
            await self._set_frontmatter_name(path, name)

        if target_path == path:
            return path

        move_response = await self.run_job(
            "move",
            src_path=path,
            dst_path=target_path,
            overwrite=False,
            retarget=True,
        )
        if not move_response.success:
            raise RuntimeError(f"move failed: {move_response.answer}")
        return target_path

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
        response_metadata = getattr(self.context.response, "metadata", None)
        if isinstance(response_metadata, dict) and response_metadata.get("modified") is False:
            self.logger.info(f"[{self.name}] result hook skipped; no resource note change modified=False")
            return
        hook = metadata.get("qwenpaw_memory_result_hook")
        if hook is None:
            return
        try:
            modified = response_metadata.get("modified") if isinstance(response_metadata, dict) else None
            self.logger.info(f"[{self.name}] result hook emit modified={modified}")
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

    async def _handle_delete(self, file_path: str, date_str: str, note_stem: str) -> None:
        daily_dir = self.config_value("daily_dir")
        fallback_path = f"{daily_dir}/{date_str}/{note_stem}.md"
        try:
            note = await self._list_resource_note(date_str, file_path, fallback_path)
        except RuntimeError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            self.logger.info(f"[{self.name}] delete list failed file_path={file_path} answer={str(exc)!r}")
            return

        note_rel = str(note["path"]) if note else fallback_path
        note_abs = self.workspace_path / note_rel
        note_existed = note_abs.is_file()
        self.logger.info(f"[{self.name}] delete start note={note_rel}")

        if note_existed:
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
            {
                "path": note_rel,
                "session_id": note_stem,
                "source_resource": self._source_resource_link(file_path),
                "action": "deleted",
                "modified": note_existed,
                "index": index_payload,
            },
        )

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
        self.logger.info(f"[{self.name}] refresh index start date={date_str} daily_dir={daily_dir}")
        index_payload = await refresh_day_index(self.file_store, date_str, daily_dir)
        self.logger.info(f"[{self.name}] refresh index done date={date_str}")

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

    async def _handle_change(self, file_path: str, raw_change) -> dict:
        assert self.context is not None
        # Handlers write item-scoped fields into the shared response. Start each
        # change with a fresh mapping so one result cannot inherit another's metadata.
        self.context.response.metadata = {}
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
            await self._handle_delete(file_path, date_str, note_stem)
        else:
            await self._handle_upsert(
                file_path,
                date_str,
                note_stem,
                change == Change.added,
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
        self.context.response.metadata["modified"] = any(
            bool((item.get("metadata") or {}).get("modified")) for item in results
        )
        await self._emit_result_hook(changes=changes, results=results)
        self.logger.info(
            f"[{self.name}] done success={success_count}/{len(changes)} "
            f"processed={len(results)} modified={self.context.response.metadata['modified']}",
        )
        return self.context.response
