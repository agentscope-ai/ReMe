"""Append a bound reminder and mark its unchanged interest notes sent."""

from .._interest import _InterestStep
from ....components import R
from ....components.codex_app_server import SessionChangedError
from ....utils.async_utils import finish_cleanup


@R.register("send_reminder_step")
class SendReminderStep(_InterestStep):
    """Deliver through the session owner, with retries limited to the live round."""

    def _files(self, value) -> list[str]:
        if not isinstance(value, list) or not value or any(not isinstance(name, str) for name in value):
            raise ValueError("files must be a nonempty list of interest filenames")
        if len(set(value)) != len(value):
            raise ValueError("files must not contain duplicates")
        for name in value:
            self._note_path(name)
        return sorted(value)

    async def _deliver(self, current: dict, message: str, files: list[str], message_id: str) -> dict:
        session_id = current["session_id"]
        snapshot = await self.agent_wrapper.read_session(session_id)
        if self._state["session_id"] != session_id or (snapshot.last_turn_id or "") != current["turn_id"]:
            return {"sent": False, "reason": "session_changed"}
        if snapshot.state.status not in {"idle", "not_loaded"}:
            return {"sent": False, "reason": "busy"}
        async with self._locked_notes(files):
            for name in files:
                post = await self._load_note(name)
                if post.get("status") != "unsent" or self._version(post) != current["versions"].get(name):
                    raise ValueError(f"Interest changed since reminder context was prepared: {name}")
            current["request"] = (message, files)
            try:
                delivered = await self.agent_wrapper.append_session(
                    session_id,
                    message,
                    message_id=message_id,
                    timeout=1,
                    expected_turn_id=current["turn_id"],
                )
            except SessionChangedError:
                return {"sent": False, "reason": "session_changed"}
            current["result"] = {"sent": True, "files": files, "message_id": message_id}
            for name in files:
                try:
                    # Re-read after the RPC; preserve any edits made during delivery.
                    post = await self._load_note(name)
                    if self._version(post) != current["versions"][name]:
                        self.logger.warning(f"Interest changed after reminder delivery; leaving unchanged: {name}")
                        continue
                    post.metadata.update(
                        status="sent",
                        sent_at=delivered.created_at or self._now().isoformat(),
                        message_id=message_id,
                        target_session_id=session_id,
                    )
                    await self._save_note(name, post)
                except (OSError, ValueError) as exc:
                    self.logger.warning(f"Reminder delivered but could not mark {name} sent: {exc}")
        return current["result"]

    async def execute(self):
        self._validate("message", "files", "session_id", "pa_round_id")
        round_id = self.context.get("pa_round_id")
        current = self._state.get("rounds", {}).get(round_id)
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
                if current["result"]["sent"]:
                    self.context.response.answer = current["result"]
                    return self.context.response
            if self._state["lock"].locked():
                self.context.response.answer = {"sent": False, "reason": "busy"}
                return self.context.response
            async with self._operation():
                current["result"] = await finish_cleanup(self._deliver(current, message, files, round_id))
                self.context.response.answer = current["result"]
        return self.context.response
