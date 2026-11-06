"""Deliver a bound reminder and recover interrupted multi-file status updates."""

import json
from uuid import uuid4

from .._interest import _InterestStep
from ...file_io._file_io import read_file_safe, write_file_safe
from ....components import R
from ....components.codex_app_server import SessionChangedError
from ....utils.async_utils import complete_in_thread, finish_cleanup


@R.register("send_reminder_step")
class SendReminderStep(_InterestStep):
    """Persist one delivery intent before asking the session owner to inject it."""

    @property
    def _pending_path(self):
        return self._path(f"{self.config_value('session_dir')}/proactive/pending.json")

    def _files(self, value) -> list[str]:
        if not isinstance(value, list) or not value or any(not isinstance(name, str) for name in value):
            raise ValueError("files must be a nonempty list of interest filenames")
        if len(set(value)) != len(value):
            raise ValueError("files must not contain duplicates")
        for name in value:
            self._note_path(name)
        return sorted(value)

    async def _delivery_notes(self, pending: dict) -> dict:
        notes = {}
        for name in self._files(pending["files"]):
            post = await self._load_note(name)
            if self._version(post, delivery=True) != pending["versions"][name]:
                raise ValueError(f"Interest changed before delivery update: {name}")
            if post.get("status") != "unsent" and not (
                post.get("status") == "sent" and post.get("message_id") == pending["message_id"]
            ):
                raise ValueError(f"Interest is no longer available: {name}")
            notes[name] = post
        return notes

    async def _deliver(self, pending: dict) -> dict:
        session_id = pending["session_id"]
        snapshot = await self.agent_wrapper.read_session(session_id)
        message = next(
            (item for item in snapshot.messages if item.id == f"reme-proactive-{pending['message_id']}"),
            None,
        )
        if message is not None and message.content != pending["message"]:
            raise ValueError("Reminder differs from its accepted delivery")
        if message is None:
            if self._state["session_id"] != session_id or (snapshot.last_turn_id or "") != pending["turn_id"]:
                await complete_in_thread(self._pending_path.unlink)
                return {"sent": False, "reason": "session_changed"}
            if snapshot.state.status not in {"idle", "not_loaded"}:
                return {"sent": False, "reason": "busy"}
        async with self._locked_notes(self._files(pending["files"])):
            await self._delivery_notes(pending)
            if message is None:
                try:
                    message = await self.agent_wrapper.append_session(
                        session_id,
                        pending["message"],
                        message_id=pending["message_id"],
                        timeout=1,
                        expected_turn_id=pending["turn_id"],
                    )
                except SessionChangedError:
                    await complete_in_thread(self._pending_path.unlink)
                    return {"sent": False, "reason": "session_changed"}
            # Re-read after the RPC; never replace user edits with a pre-RPC snapshot.
            notes = await self._delivery_notes(pending)
            for name, post in notes.items():
                post.metadata.update(
                    status="sent",
                    sent_at=message.created_at or self._now().isoformat(),
                    message_id=pending["message_id"],
                    target_session_id=session_id,
                )
                await self._save_note(name, post)
        await complete_in_thread(self._pending_path.unlink)
        return {"sent": True, "files": pending["files"], "message_id": pending["message_id"]}

    async def recover(self) -> dict | None:
        """Resolve a previous intent before generating another reminder."""
        if not self._pending_path.exists():
            return None
        if self._state["lock"].locked():
            return {"sent": False, "reason": "busy"}
        async with self._operation():
            content, _ = await read_file_safe(
                self._pending_path,
                max_bytes=self.max_file_bytes(),
                require_complete=True,
            )
            return await finish_cleanup(self._deliver(json.loads(content)))

    async def execute(self):
        self._validate("message", "files", "session_id", "pa_round_id")
        current = self._state.get("rounds", {}).get(self.context.get("pa_round_id"))
        if current is None or self.context.get("session_id") != current["session_id"]:
            raise ValueError("send_reminder requires a live, server-bound proactive round")
        message = self.context.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 100_000:
            raise ValueError("message must contain 1-100000 characters")
        files = self._files(self.context.get("files"))
        async with current["lock"]:
            request = (message, files)
            if current["request"] is not None:
                if request != current["request"]:
                    raise ValueError("Only one reminder may be requested per round")
                result = await self.recover()
                self.context.response.answer = current["result"] if result is None else result
                current["result"] = self.context.response.answer
                return self.context.response
            if self._state["lock"].locked():
                self.context.response.answer = {"sent": False, "reason": "busy"}
                return self.context.response
            async with self._operation():
                if self._pending_path.exists():
                    raise RuntimeError("An earlier reminder must be recovered first")
                if self._state["session_id"] != current["session_id"]:
                    self.context.response.answer = {"sent": False, "reason": "session_changed"}
                    return self.context.response
                async with self._locked_notes(files):
                    notes = {name: await self._load_note(name) for name in files}
                    for name, post in notes.items():
                        if post.get("status") != "unsent" or self._version(post) != current["versions"].get(name):
                            raise ValueError(f"Interest changed since reminder context was prepared: {name}")
                    pending = {
                        "message_id": uuid4().hex,
                        "session_id": current["session_id"],
                        "turn_id": current["turn_id"],
                        "message": message,
                        "files": files,
                        "versions": {name: self._version(post, delivery=True) for name, post in notes.items()},
                    }
                    await write_file_safe(self._pending_path, json.dumps(pending, ensure_ascii=False))
                current["request"] = request
                current["result"] = await finish_cleanup(self._deliver(pending))
                self.context.response.answer = current["result"]
        return self.context.response
