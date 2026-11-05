"""Choose a contextual reminder and deliver it through the session owner."""

from uuid import uuid4

import frontmatter

from ._interest import _InterestStep
from ..file_io._file_io import get_path_lock
from ...components import R
from ...components.codex_app_server import SessionChangedError


@R.register("push_interest_step")
class PushInterestStep(_InterestStep):
    """Keep one retry-stable delivery and recheck conversation context under its lease."""

    async def _deliver(self, path: str, post: frontmatter.Post) -> bool:
        session_id = post["target_session_id"]
        message_id = post["message_id"]
        if post.get("delivery_attempted_at"):
            snapshot = await self.agent_wrapper.read_session(session_id)
            delivered = next((m for m in snapshot.messages if m.id == f"reme-proactive-{message_id}"), None)
            if delivered is not None:
                if delivered.content != post["message"]:
                    raise ValueError("Interest message differs from its accepted delivery")
                await self._update_note(path, post, status="sent", sent_at=delivered.created_at, last_error=None)
                return True
        if self._state["lock"].locked():
            return False
        async with self._operation():
            if self._state["session_id"] != session_id:
                await self._update_note(path, post, status="unsent", delivery_attempted_at=None)
                return False
            lock = await get_path_lock(self._path(path))
            async with lock:
                current = await self._load_note(path)
                if frontmatter.dumps(current) != frontmatter.dumps(post):
                    raise ValueError("Interest changed before delivery")
                current["delivery_attempted_at"] = self._now().isoformat()
                await self._save_note(path, current)
                try:
                    message = await self.agent_wrapper.append_session(
                        session_id,
                        current["message"],
                        message_id=message_id,
                        timeout=1,
                        expected_turn_id=current["context_turn_id"],
                    )
                except SessionChangedError:
                    current.metadata.update(status="unsent", delivery_attempted_at=None)
                except Exception as exc:
                    current["last_error"] = str(exc)
                    await self._save_note(path, current)
                    raise
                else:
                    current.metadata.update(status="sent", sent_at=message.created_at, last_error=None)
                await self._save_note(path, current)
                return current["status"] == "sent"

    async def execute(self):
        memory = await self._memory()
        before = await self._notes()
        pending = {path: post for path, post in before.items() if post["status"] in {"unsent", "ready"}}
        ready = [(path, post) for path, post in pending.items() if post["status"] == "ready"]
        if len(ready) > 1:
            raise ValueError("Only one interest may be ready for delivery")
        if ready and ready[0][1].get("delivery_attempted_at"):
            self.context.response.answer = {"sent": await self._deliver(*ready[0])}
            return self.context.response
        session_id = self._state["session_id"]
        if not pending or session_id is None:
            self.context.response.answer = {"sent": False}
            return self.context.response
        snapshot = await self.agent_wrapper.read_session(session_id)
        status = "active" if self._state["lock"].locked() else snapshot.state.status
        await self._agent(
            memory,
            {
                "current_session_id": session_id,
                "current_session_file": snapshot.path,
                "current_session_status": status,
                "last_turn_id": snapshot.last_turn_id,
                "recent_messages": [
                    {"role": m.role, "content": m.content, "source": m.source, "completed": m.completed}
                    for m in snapshot.messages[-20:]
                ],
                "interests": self._summaries(before),
            },
            ["set_next_wakeup"],
        )
        notes = await self._notes(before)
        if notes.keys() - before.keys():
            raise ValueError("Push agent must not create interests")
        for path, original in before.items():
            current = notes.get(path)
            if current is None:
                raise ValueError(f"Push agent removed an interest: {path}")
            editable = {"status", "message", "updated_at"}
            if (
                current.content != original.content
                or {key: value for key, value in current.metadata.items() if key not in editable}
                != {key: value for key, value in original.metadata.items() if key not in editable}
                or (original["status"] in {"unsent", "ready"} and current["status"] == "sent")
            ):
                raise ValueError(f"Push agent changed protected interest fields: {path}")
        ready = [(path, post) for path, post in notes.items() if post["status"] == "ready"]
        if len(ready) > 1:
            raise ValueError("Only one interest may be ready for delivery")
        sent = False
        if ready:
            path, post = ready[0]
            message = post.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > 100_000:
                raise ValueError("A ready interest must contain a message of 1-100000 characters")
            if status not in {"idle", "not_loaded"}:
                await self._update_note(path, post, status="unsent")
            else:
                post = await self._update_note(
                    path,
                    post,
                    target_session_id=session_id,
                    context_turn_id=snapshot.last_turn_id or "",
                    message_id=uuid4().hex,
                )
                sent = await self._deliver(path, post)
        self.context.response.answer = {"sent": sent}
        return self.context.response
