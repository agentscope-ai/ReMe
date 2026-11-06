"""Run Remy's collector and initialize its optional output note."""

from ._interest import _InterestStep
from ...components import R


@R.register("find_interest_step")
class FindInterestStep(_InterestStep):
    """Leave resource selection and interest learning to the collector prompt."""

    async def execute(self):
        self._validate("default_wakeup_minutes")
        self.context.response.answer = {"created": []}
        await self._ensure_memory()
        session_id = self._state["session_id"]
        if session_id is None:
            return self.context.response
        session_path = await self.agent_wrapper.get_session_path(session_id)
        if session_id != self._state["session_id"]:
            return self.context.response
        now = self._now()
        filename, suffix = f"interest_{now:%H%M%S}.md", 1
        while self._note_path(filename).exists():
            filename, suffix = f"interest_{now:%H%M%S}_{suffix}.md", suffix + 1
        await self._agent(
            ["resource_changes", "resource_processed"],
            session_id=session_id,
            session_path=session_path or "",
            output_path=self._note_path(filename),
        )
        async with self._locked_notes([filename]):
            if self._note_path(filename).exists():
                post = await self._load_note(filename)
                if post.content.strip():
                    post.metadata.update(status="unsent", created_at=now.isoformat())
                    await self._save_note(filename, post)
                    self.context.response.answer["created"] = [
                        self._note_path(filename).relative_to(self.workspace_path).as_posix(),
                    ]
        return self.context.response
