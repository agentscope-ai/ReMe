"""Produce a final, evidence-backed answer for a LongMemEval case.

The step puts the complete query, golden-answer object, and any available
disputed reference answers directly into the prompt.  Raw session content stays out of the model
context: Claude Code starts in the sample's ``session`` directory and uses its
normal file tools to inspect whichever sessions it needs. Session timestamps
are scanned only to identify evidence that did not exist at question time;
``answer_session_ids`` are not evaluated.

Claude Code is intentionally used without an output schema.  Its ordinary text
reply may contain narration but must include exactly one fenced ``json`` block
whose object contains ``reason``, ``golden_answer_correct``, ``answer``, and
``is_session_time_wrong``.  API errors and invalid replies are retried with
capped exponential backoff.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ....components import R
from ...base_step import BaseStep

DEFAULT_REFERENCE_PATHS = (
    "benchmark/longmemeval/golden_check_list_false.jsonl",
    "benchmark/longmemeval/merge_confirm_jinli_false.jsonl",
)
REFERENCE_PATHS_ENV = "LME_FINAL_ANSWER_REFERENCE_PATHS"
RETRY_INITIAL_SECONDS = 5.0
RETRY_MAX_SECONDS = 300.0
_LME_DATETIME_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")
_FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


@R.register("lme_final_answer_review_step")
class FinalAnswerReviewStep(BaseStep):
    """Ask a Claude Code agent to review one golden answer."""

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as file:
                value = json.load(file)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read LongMemEval file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in LongMemEval file: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return value

    @staticmethod
    def _parse_datetime(raw_date: Any, *, source: str) -> datetime:
        text = str(raw_date or "").strip()
        match = _LME_DATETIME_RE.search(text)
        if match is None:
            raise ValueError(f"Invalid LongMemEval datetime in {source}: {text!r}")
        try:
            return datetime(*(int(part) for part in match.groups()))
        except ValueError as exc:
            raise ValueError(
                f"Invalid LongMemEval datetime in {source}: {text!r}",
            ) from exc

    def _resolve_reference_path(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path

        # The configured defaults are repository-relative.  Tests and custom
        # jobs may instead provide workspace-relative fixture paths.
        repository_path = Path.cwd() / path
        if repository_path.is_file():
            return repository_path
        return self.workspace_path / path

    def _load_references(self, question_id: str) -> list[dict[str, Any]]:
        raw_paths: Any
        serialized_paths = os.environ.get(REFERENCE_PATHS_ENV)
        if serialized_paths:
            try:
                raw_paths = json.loads(serialized_paths)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{REFERENCE_PATHS_ENV} must be a JSON array of paths") from exc
        else:
            raw_paths = self.kwargs.get("reference_paths") or DEFAULT_REFERENCE_PATHS
        if isinstance(raw_paths, str):
            raw_paths = [raw_paths]
        if not isinstance(raw_paths, (list, tuple)) or not raw_paths:
            raise ValueError("reference_paths must contain at least one JSONL path")

        references: list[dict[str, Any]] = []
        for raw_path in raw_paths:
            path = self._resolve_reference_path(str(raw_path))
            try:
                with path.open(encoding="utf-8") as file:
                    for line_number, line in enumerate(file, start=1):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSONL at {path}:{line_number}",
                            ) from exc
                        if not isinstance(item, dict):
                            raise ValueError(
                                f"Expected a JSON object at {path}:{line_number}",
                            )
                        if str(item.get("question_id") or "") == question_id:
                            references.append({"source": path.name, **item})
            except OSError as exc:
                raise FileNotFoundError(
                    f"Cannot read reference-answer file: {path}",
                ) from exc

        return references

    def _inspect_session_times(self, question_dt: datetime) -> tuple[int, list[dict[str, str]]]:
        """Return the session count and timestamp-only metadata for future sessions."""
        resource_dir = self.app_context.app_config.resource_dir if self.app_context is not None else "session"
        session_dir = self.workspace_path / resource_dir
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

        session_paths = sorted(session_dir.glob("*.json"))
        future_sessions: list[dict[str, str]] = []
        for path in session_paths:
            session = self._load_json(path)
            session_id = str(session.get("haystack_session_id") or path.stem)
            session_date = str(session.get("haystack_date") or "").strip()
            session_dt = self._parse_datetime(
                session_date,
                source=f"{path}:haystack_date",
            )
            if session_dt > question_dt:
                future_sessions.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "session_file": path.name,
                    },
                )
        return len(session_paths), future_sessions

    @staticmethod
    def _parse_reply(raw_reply: Any) -> dict[str, Any]:
        if not isinstance(raw_reply, str) or not raw_reply.strip():
            raise ValueError("Agent returned an empty reply")
        json_blocks = _FENCED_JSON_RE.findall(raw_reply)
        if len(json_blocks) != 1:
            raise ValueError("Agent reply must contain exactly one fenced ```json``` block")
        try:
            value = json.loads(json_blocks[0].strip())
        except json.JSONDecodeError as exc:
            raise ValueError("Agent's fenced json block is not valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Agent reply must be a JSON object")
        if set(value) != {"reason", "golden_answer_correct", "answer", "is_session_time_wrong"}:
            raise ValueError(
                "Agent reply must contain exactly 'reason', 'golden_answer_correct', 'answer', "
                "and 'is_session_time_wrong'",
            )
        answer = value["answer"]
        reason = value["reason"]
        golden_answer_correct = value["golden_answer_correct"]
        is_session_time_wrong = value["is_session_time_wrong"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Agent reply 'reason' must be a non-empty string")
        if "answer_session_ids" in reason.casefold():
            raise ValueError("Agent reply 'reason' must not evaluate answer_session_ids")
        if not isinstance(golden_answer_correct, bool):
            raise ValueError("Agent reply 'golden_answer_correct' must be a boolean")
        if not isinstance(answer, str):
            raise ValueError("Agent reply 'answer' must be a string")
        answer = answer.strip()
        if golden_answer_correct and answer:
            raise ValueError("Agent reply 'answer' must be empty when golden_answer_correct is true")
        if not golden_answer_correct and not answer:
            raise ValueError("Agent reply 'answer' must be non-empty when golden_answer_correct is false")
        if not isinstance(is_session_time_wrong, bool):
            raise ValueError("Agent reply 'is_session_time_wrong' must be a boolean")
        if is_session_time_wrong:
            raise ValueError("Agent reply 'is_session_time_wrong' is deprecated and must be false")
        return {
            "reason": reason.strip(),
            "golden_answer_correct": golden_answer_correct,
            "answer": answer,
            "is_session_time_wrong": is_session_time_wrong,
        }

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_final_answer_review_step requires agent_wrapper")

        query = self._load_json(self.workspace_path / "query.json")
        golden = self._load_json(self.workspace_path / "answer.json")
        question_id = str(query.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("query.json requires a non-empty 'question_id'")
        question_dt = self._parse_datetime(
            query.get("question_date"),
            source="query.json:question_date",
        )
        references = self._load_references(question_id)
        num_sessions, future_sessions = self._inspect_session_times(question_dt)

        payload = {
            "query": query,
            "answer_json": golden,
            "reference_answers": references,
            "session_time_check": {
                "sessions_after_question_date": future_sessions,
            },
        }
        user_prompt = self.prompt_format(
            "user_message",
            question_id=question_id,
            question_date=str(query.get("question_date") or ""),
            num_sessions=num_sessions,
            num_future_sessions=len(future_sessions),
            num_references=len(references),
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        retry_initial_seconds = float(
            self.kwargs.get("retry_initial_seconds", RETRY_INITIAL_SECONDS),
        )
        retry_max_seconds = float(
            self.kwargs.get("retry_max_seconds", RETRY_MAX_SECONDS),
        )
        if retry_initial_seconds <= 0:
            retry_initial_seconds = RETRY_INITIAL_SECONDS
        retry_max_seconds = max(retry_max_seconds, retry_initial_seconds)

        attempt = 1
        sleep_seconds = retry_initial_seconds
        while True:
            try:
                # Deliberately do not pass output_schema: this case evaluates an
                # ordinary Claude Code response and validates it afterward.
                result = await self.agent_wrapper.reply(
                    user_prompt,
                    system_prompt=self.get_prompt("system_prompt"),
                )
                final_answer = self._parse_reply(result.get("result"))
                if attempt > 1:
                    self.logger.info(
                        f"[{self.name}] recovered after {attempt} attempts",
                    )
                break
            except Exception as exc:  # noqa: BLE001 - agent/API/format failures share the retry contract
                delay = min(sleep_seconds, retry_max_seconds)
                self.logger.warning(
                    f"[{self.name}] attempt {attempt} failed for {question_id}: {exc}; retrying in {delay:.1f}s",
                )
                await asyncio.sleep(delay)
                sleep_seconds = min(sleep_seconds * 2, retry_max_seconds)
                attempt += 1

        self.context.response.success = True
        self.context.response.answer = json.dumps(final_answer, ensure_ascii=False)
        self.context.response.metadata.update(
            {
                "question_id": question_id,
                "num_sessions": num_sessions,
                "num_future_sessions": len(future_sessions),
                "future_sessions": future_sessions,
                "num_reference_answers": len(references),
                "is_session_time_wrong": False,
                "attempts": attempt,
                "agent_session_id": result.get("session_id"),
            },
        )
        return self.context.response
