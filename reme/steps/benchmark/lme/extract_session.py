"""extract_session_by_id — deep-read one raw session, keyed by its session_id.

This is the hand-written function tool the answering agent sees. Search results
surface a note's ``session_id``; when a hit looks relevant, the agent passes that
``session_id`` here. The step resolves the question/time from ``query.json``,
locates the raw session file (named ``<date>_(...)_<time>@<session_id>.json``
under ``resource_dir``), loads its messages, and asks an agent to extract —
completely and verbatim — every part of that session relevant to the question.
"""

import json
from pathlib import Path

from ...base_step import BaseStep
from ....components import R


@R.register("lme_extract_session_step")
class LmeExtractSessionStep(BaseStep):
    """Resolve a session_id to raw content, then deep-read it for the question."""

    def _resource_dir_name(self) -> str:
        return self.app_context.app_config.resource_dir if self.app_context is not None else "session"

    def _load_query(self) -> dict:
        path = self.workspace_path / "query.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("query.json is not a JSON object")
        return data

    def _find_session_file(self, session_id: str) -> Path | None:
        session_dir = self.workspace_path / self._resource_dir_name()
        if not session_dir.is_dir():
            return None
        # Files are named "<date>_(...)_<time>@<session_id>.json".
        matches = list(session_dir.glob(f"*@{session_id}.json"))
        if matches:
            return matches[0]
        # Fall back to a plain "<session_id>.json" naming.
        direct = session_dir / f"{session_id}.json"
        return direct if direct.is_file() else None

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_extract_session_step requires agent_wrapper")

        session_id: str = str(self.context.get("session_id", "") or "").strip()
        if not session_id:
            self.context.response.success = False
            self.context.response.answer = "Error: session_id is required"
            return self.context.response

        try:
            query = self._load_query()
        except (OSError, ValueError) as exc:
            self.context.response.success = False
            self.context.response.answer = f"Error: cannot read query.json: {exc}"
            return self.context.response
        question = str(query.get("question", "") or "").strip()
        question_time = str(query.get("question_date", "") or "").strip()

        session_path = self._find_session_file(session_id)
        if session_path is None:
            self.context.response.success = False
            self.context.response.answer = (
                f"Error: no session file found for session_id={session_id!r}. "
                "Use a session_id shown in a search result."
            )
            return self.context.response

        try:
            with session_path.open(encoding="utf-8") as f:
                session = json.load(f)
        except (OSError, ValueError) as exc:
            self.context.response.success = False
            self.context.response.answer = f"Error: cannot read session {session_path.name}: {exc}"
            return self.context.response

        messages = session.get("messages") if isinstance(session, dict) else None
        session_content = json.dumps(messages or session, ensure_ascii=False, indent=2)

        user_prompt = self.prompt_format(
            "user_message",
            question=question or "(unknown)",
            question_time=question_time or "(unknown)",
            session_content=session_content,
        )
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.get_prompt("system_prompt"),
        )
        answer = (result.get("result") or "").strip()

        self.logger.info(f"[{self.name}] extracted {len(answer)} chars for session_id={session_id!r}")
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {"session_id": session_id, "session_file": session_path.name},
        )
        return self.context.response
