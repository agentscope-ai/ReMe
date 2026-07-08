"""Review every LongMemEval session for query/answer-relevant evidence.

For a workspace such as ``datasets/longmemeval/1`` this step loads ``query.json``
and ``answer.json``, then walks each session under ``resource_dir`` one by one.
An agent wrapper extracts all information relevant to the question or the golden
answer (time information in particular), and — when a session is one of the
``answer_session_ids`` — sanity-checks whether that attribution is plausible
(e.g. an answer session dated *after* the question is contradictory).

The collected per-session summaries are stashed on the context under
``session_summaries`` for the downstream golden-answer check.
"""

import asyncio
import json
from pathlib import Path

from ...base_step import BaseStep
from ....components import R

# Default number of sessions reviewed concurrently; override via the step's
# ``concurrency`` config field.
DEFAULT_CONCURRENCY = 8

# Structured schema the review agent must return for each session.
_SESSION_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {
            "type": "boolean",
            "description": "Whether this session contains anything relevant to the question or golden answer.",
        },
        "relevant_info": {
            "type": "string",
            "description": "Complete extraction of every piece of information relevant to the question or "
            "answer. Leave empty when the session is irrelevant.",
        },
        "time_info": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every time/date reference relevant to the question or answer, extracted verbatim "
            "with the fact it is attached to (e.g. 'started the job on 2023/03/01').",
        },
        "is_answer_session": {
            "type": "boolean",
            "description": "Whether this session's id appears in answer_session_ids.",
        },
        "answer_session_check": {
            "type": "string",
            "description": "Only when is_answer_session is true: judge whether the attribution is reasonable "
            "(e.g. does this session actually support the golden answer? is its date not later than the "
            "question date?). Use 'N/A' when this is not an answer session.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any contradictions or problems found, such as a date conflict or an answer session "
            "that cannot support the golden answer.",
        },
    },
    "required": ["is_relevant", "relevant_info", "time_info", "is_answer_session", "answer_session_check", "issues"],
}


@R.register("lme_session_review_step")
class SessionReviewStep(BaseStep):
    """Extract query/answer-relevant evidence from every session, serially."""

    def _load_json(self, filename: str) -> dict:
        path = self.workspace_path / filename
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read LongMemEval file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LongMemEval file: {path}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return data

    def _session_dir(self) -> Path:
        resource_dir = self.app_context.app_config.resource_dir if self.app_context is not None else "session"
        return self.workspace_path / resource_dir

    def _concurrency(self) -> int:
        """Number of sessions to review concurrently (``concurrency`` config, default 8)."""
        raw = self.context.get("concurrency") if self.context is not None else None
        if raw is None:
            raw = self.kwargs.get("concurrency", DEFAULT_CONCURRENCY)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_CONCURRENCY
        return value if value >= 1 else DEFAULT_CONCURRENCY

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_session_review_step requires agent_wrapper")

        query_data = self._load_json("query.json")
        answer_data = self._load_json("answer.json")

        question = str(query_data.get("question") or "").strip()
        question_type = str(query_data.get("question_type") or "").strip()
        question_date = str(query_data.get("question_date") or "").strip()
        if not question:
            raise ValueError("query.json requires a non-empty 'question'")

        golden_answer = str(answer_data.get("answer") or "").strip()
        answer_session_ids = [str(s) for s in (answer_data.get("answer_session_ids") or [])]

        session_dir = self._session_dir()
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        session_files = sorted(p for p in session_dir.iterdir() if p.suffix == ".json")

        concurrency = self._concurrency()
        total = len(session_files)
        self.logger.info(
            f"[{self.name}] reviewing {total} sessions from {session_dir} (concurrency={concurrency})",
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def review_one(idx: int, session_path: Path) -> dict | None:
            try:
                session = self._load_json_path(session_path)
            except (ValueError, FileNotFoundError) as exc:
                self.logger.warning(f"[{self.name}] skip {session_path.name}: {exc}")
                return None

            session_id = str(session.get("haystack_session_id") or session_path.stem)
            session_date = str(session.get("haystack_date") or "").strip()
            is_answer_session = session_id in answer_session_ids

            user_prompt = self.prompt_format(
                "user_message",
                question=question,
                question_type=question_type,
                question_date=question_date,
                golden_answer=golden_answer,
                answer_session_ids=", ".join(answer_session_ids) or "(none)",
                session_id=session_id,
                session_date=session_date,
                session_content=json.dumps(session.get("messages", []), ensure_ascii=False, indent=2),
            )
            async with semaphore:
                try:
                    result = await self.agent_wrapper.reply(
                        user_prompt,
                        system_prompt=self.get_prompt("system_prompt"),
                        output_schema=_SESSION_REVIEW_SCHEMA,
                    )
                except Exception as exc:  # noqa: BLE001 — one bad session must not abort the sweep
                    self.logger.warning(f"[{self.name}] review failed for {session_id}: {exc}")
                    return None

            extracted = result.get("structured_output")
            if not isinstance(extracted, dict):
                # Fall back to the free-text reply when structured output is unavailable.
                extracted = {"relevant_info": (result.get("result") or "").strip()}

            summary = {
                "session_id": session_id,
                "session_date": session_date,
                "is_answer_session": is_answer_session,
                **extracted,
            }
            self.logger.info(
                f"[{self.name}] ({idx}/{total}) {session_id} "
                f"relevant={summary.get('is_relevant')} answer_session={is_answer_session}",
            )
            return summary

        # gather preserves input order, so summaries stay chronological.
        results = await asyncio.gather(
            *(review_one(idx, path) for idx, path in enumerate(session_files, start=1)),
        )
        summaries: list[dict] = [s for s in results if s is not None]

        self.context["query_data"] = query_data
        self.context["answer_data"] = answer_data
        self.context["question"] = question
        self.context["question_type"] = question_type
        self.context["question_date"] = question_date
        self.context["golden_answer"] = golden_answer
        self.context["answer_session_ids"] = answer_session_ids
        self.context["session_summaries"] = summaries

        self.context.response.success = True
        self.context.response.answer = f"reviewed {len(summaries)} sessions"
        self.context.response.metadata.update(
            {
                "question": question,
                "question_date": question_date,
                "answer_session_ids": answer_session_ids,
                "num_sessions": len(summaries),
                "session_summaries": summaries,
            },
        )
        return self.context.response

    def _load_json_path(self, path: Path) -> dict:
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            raise FileNotFoundError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if not isinstance(data, dict):
            raise ValueError("session file is not a JSON object")
        return data
