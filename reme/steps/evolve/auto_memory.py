"""auto_memory — record conversation facts into a daily note via an agent."""

from pathlib import Path

import aiofiles
from agentscope.message import Msg

from ._evolve import format_history, now
from ..base_step import BaseStep
from ..file_io import refresh_day_index, validate_session_id
from ...components import R

_TOOL_OUTPUT_MAX = 2048
_TOOL_OUTPUT_HALF = 1024
_SOURCE_CONVERSATION_KEY = "source_conversation"


def _truncate_text(text: str) -> str:
    if len(text) <= _TOOL_OUTPUT_MAX:
        return text
    return text[:_TOOL_OUTPUT_HALF] + "\n...(truncated)...\n" + text[-_TOOL_OUTPUT_HALF:]


def _sanitize_tool_result(block):
    output = block.output
    if isinstance(output, str):
        truncated = _truncate_text(output)
        if truncated is output:
            return block
        return block.model_copy(update={"output": truncated})
    new_output = []
    changed = False
    for item in output:
        if item.type == "data":
            changed = True
            continue
        if item.type == "text":
            truncated = _truncate_text(item.text)
            if truncated is not item.text:
                changed = True
                new_output.append(item.model_copy(update={"text": truncated}))
            else:
                new_output.append(item)
        else:
            new_output.append(item)
    if not changed:
        return block
    return block.model_copy(update={"output": new_output})


def _sanitize_msg_for_save(msg: Msg) -> Msg:
    new_content = []
    changed = False
    for block in msg.content:
        if block.type == "data" and hasattr(block, "source") and getattr(block.source, "type", None) == "base64":
            changed = True
            continue
        if block.type == "tool_result":
            sanitized = _sanitize_tool_result(block)
            if sanitized is not block:
                changed = True
            new_content.append(sanitized)
        else:
            new_content.append(block)
    if not changed:
        return msg
    return msg.model_copy(update={"content": new_content})


@R.register("auto_memory_step")
class AutoMemoryStep(BaseStep):
    """Record conversation facts into a daily note via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    def _session_dir(self) -> str:
        return str(self.config_value("session_dir")).strip("/")

    def _session_path(self, session_id: str) -> Path:
        return self.file_store.workspace_path / self._session_dir() / "dialog" / f"{session_id}.jsonl"

    def _session_link(self, session_id: str) -> str:
        return f"[[{self._session_dir()}/dialog/{session_id}.jsonl]]"

    async def _save_session_messages(self, session_id: str, messages: list[Msg]) -> None:
        if not session_id or not messages:
            return

        path = self._session_path(session_id)

        existing: list[Msg] = []
        if path.exists():
            async with aiofiles.open(path, encoding="utf-8") as f:
                content = await f.read()
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        existing.append(Msg.model_validate_json(line))
                    except Exception:
                        pass

        by_id: dict[str, Msg] = {}
        for msg in existing:
            by_id[msg.id] = msg
        for msg in messages:
            by_id[msg.id] = msg
        merged = sorted(by_id.values(), key=lambda m: m.created_at)

        can_append = 0 < len(existing) <= len(merged) and all(
            merged[i].id == existing[i].id for i in range(len(existing))
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        if can_append:
            new_msgs = merged[len(existing) :]
            if new_msgs:
                async with aiofiles.open(path, "a", encoding="utf-8") as f:
                    for msg in new_msgs:
                        await f.write(_sanitize_msg_for_save(msg).model_dump_json() + "\n")
        else:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                for msg in merged:
                    await f.write(_sanitize_msg_for_save(msg).model_dump_json() + "\n")

    @staticmethod
    def _to_msg(item) -> Msg:
        if isinstance(item, Msg):
            return item
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            item = {**item, "content": [{"type": "text", "text": item["content"]}]}
        return Msg.model_validate(item)

    async def execute(self):
        assert self.context is not None
        raw_messages = self.context.get("messages") or []
        session_id: str = self.context.get("session_id", "")
        memory_hint: str = self.context.get("memory_hint", "")
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        current = now(tz)

        messages: list[Msg] = [self._to_msg(item) for item in raw_messages]

        if session_id and (err := validate_session_id(session_id)):
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            return

        await self._save_session_messages(session_id, messages)

        if not messages:
            self.context.response.success = True
            self.context.response.answer = "Skipped: no messages"
            self.context.response.metadata.update({"n_messages": 0})
            self.logger.info(f"[{self.name}] Skipped: no messages session_id={session_id!r}")
            return

        create_response = await self.run_job("daily_create", session_id=session_id)
        if not create_response.success:
            self.context.response.success = False
            self.context.response.answer = f"daily_create failed: {create_response.answer}"
            self.logger.info(f"[{self.name}] daily_create failed session_id={session_id!r}")
            return

        note_path: str = create_response.metadata["path"]
        created: bool = create_response.metadata["created"]
        self.logger.info(f"[{self.name}] {note_path} created={created} msgs={len(messages)} hint={bool(memory_hint)}")

        template_key = "user_message_create" if created else "user_message_update"
        user_message = self.prompt_format(
            template_key,
            today=current.strftime("%Y-%m-%d"),
            workspace_dir=str(self.file_store.workspace_path),
            note=memory_hint or "(none)",
            note_path=note_path,
            history=format_history(messages),
        )

        result = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            job_tools=self.agent_tools,
        )

        source_conversation = ""
        if session_id:
            source_conversation = self._session_link(session_id)
            link_response = await self.run_job(
                "frontmatter_update",
                path=note_path,
                metadata={_SOURCE_CONVERSATION_KEY: source_conversation},
            )
            if not link_response.success:
                self.context.response.success = False
                self.context.response.answer = f"frontmatter_update failed: {link_response.answer}"
                self.context.response.metadata.update(
                    {"path": note_path, "created": created, "n_messages": len(messages), "index": None},
                )
                self.logger.info(
                    f"[{self.name}] source conversation link failed "
                    f"path={note_path} session_id={session_id!r} answer={link_response.answer!r}",
                )
                return

        daily_dir = self.config_value("daily_dir")
        index_payload = await refresh_day_index(self.file_store, create_response.metadata["date"], daily_dir)

        self.context.response.success = True
        self.context.response.answer = (result.get("result") or "").strip()
        self.context.response.metadata.update(
            {
                "path": note_path,
                "created": created,
                "n_messages": len(messages),
                "source_conversation": source_conversation,
                "index": index_payload,
            },
        )
        self.logger.info(f"[{self.name}] done {note_path}")
