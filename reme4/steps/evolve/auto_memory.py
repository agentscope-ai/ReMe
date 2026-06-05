"""auto_memory — record conversation facts into a daily note via an agent."""

from pathlib import Path

import aiofiles
from agentscope.message import Msg

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R


@R.register("auto_memory_step")
class AutoMemoryStep(BaseStep):
    """Record conversation facts into a daily note via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    def _session_path(self, session_id: str, tz: str | None) -> Path:
        current = now(tz)
        date_str = current.strftime("%Y-%m-%d")
        resource = self.app_context.app_config.resource_dir if self.app_context else "resource"
        return self.file_store.vault_path / resource / date_str / f"session_agent_{session_id}.jsonl"

    async def _save_session_messages(self, session_id: str, messages: list[Msg], tz: str | None) -> None:
        if not session_id or not messages:
            return

        path = self._session_path(session_id, tz)

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
                        await f.write(msg.model_dump_json() + "\n")
        else:
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                for msg in merged:
                    await f.write(msg.model_dump_json() + "\n")

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

        await self._save_session_messages(session_id, messages, tz)

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
            vault_dir=str(self.file_store.vault_path),
            note=memory_hint or "(none)",
            note_path=note_path,
            history=format_history(messages),
        )

        tools = [self.get_job(name) for name in self.agent_tools]
        _, msg = await self.agent_wrapper.reply(
            user_message,
            system_prompt=self.prompt_format("system_prompt"),
            tools=tools,
        )

        self.context.response.success = True
        self.context.response.answer = (msg.get_text_content() or "").strip()
        self.context.response.metadata.update(
            {"path": note_path, "created": created, "n_messages": len(messages)},
        )
        self.logger.info(f"[{self.name}] done {note_path}")
