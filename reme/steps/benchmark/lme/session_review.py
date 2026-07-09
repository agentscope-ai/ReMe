"""Review every LongMemEval session for query/answer-relevant evidence.

For a workspace such as ``datasets/longmemeval/1`` this step loads ``query.json``
and ``answer.json``, filters out sessions dated after ``question_date``, then
walks each remaining session under ``resource_dir`` one by one. An agent wrapper
extracts all information relevant to the question or the golden answer, keeping
time information attached to the facts it modifies.

The collected per-session summaries are written to ``session_review.json`` for
the downstream golden-answer check.
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

from ...base_step import BaseStep
from ....components import R

START_INTERVAL_SECONDS = 1.0
OUTPUT_FILENAME = "session_review.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
GLOBAL_THROTTLE_PATH = REPO_ROOT / "logs" / "session_review" / ".submit_throttle"
_LME_DATETIME_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")

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
            "description": "Exhaustive extraction of every detail relevant to the question or golden answer. "
            "Do not omit relevant facts, names, entities, numbers, preferences, corrections, or constraints. "
            "Include every relevant time/date reference inline with the fact it modifies "
            "(e.g. 'started the job on 2023/03/01'). Leave empty when the session is irrelevant.",
        },
    },
    "required": ["is_relevant", "relevant_info"],
    "additionalProperties": False,
}


@R.register("lme_session_review_step")
class SessionReviewStep(BaseStep):
    """Extract query/answer-relevant evidence from every eligible session."""

    def _load_json(self, path: Path | str) -> dict:
        if not isinstance(path, Path):
            path = self.workspace_path / path
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

    @staticmethod
    def _parse_lme_datetime(raw_date: str) -> datetime | None:
        """Parse LongMemEval timestamps like ``2023/05/20 (Sat) 03:29``."""
        match = _LME_DATETIME_RE.search(raw_date.strip())
        if match is None:
            return None
        try:
            year, month, day, hour, minute = (int(part) for part in match.groups())
            return datetime(year, month, day, hour, minute)
        except ValueError:
            return None

    @staticmethod
    def _wait_for_global_start_slot() -> None:
        """Throttle request starts across all concurrent ``session_review`` processes."""
        try:
            import portalocker
        except ImportError as exc:
            raise RuntimeError(
                "lme_session_review_step requires the benchmark extra for cross-process throttling. "
                "Install with `pip install 'reme-ai[benchmark]'`.",
            ) from exc

        GLOBAL_THROTTLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(GLOBAL_THROTTLE_PATH, mode="a+", encoding="utf-8") as f:
            f.seek(0)
            raw = f.read().strip()
            try:
                last_started_at = float(raw) if raw else 0.0
            except ValueError:
                last_started_at = 0.0

            now = time.time()
            next_start_at = last_started_at + START_INTERVAL_SECONDS
            if now < next_start_at:
                time.sleep(next_start_at - now)
                now = time.time()

            f.seek(0)
            f.truncate()
            f.write(f"{now:.6f}")
            f.flush()

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
        question_dt = self._parse_lme_datetime(question_date)
        if question_dt is None:
            raise ValueError(f"query.json has an invalid 'question_date': {question_date!r}")

        golden_answer = str(answer_data.get("answer") or "").strip()
        answer_session_ids = [str(s) for s in (answer_data.get("answer_session_ids") or [])]

        session_dir = self._session_dir()
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        session_files = sorted(p for p in session_dir.iterdir() if p.suffix == ".json")
        sessions: list[tuple[dict, str, str]] = []
        filtered_sessions: list[dict] = []
        session_ids_illegal: list[str] = []
        answer_session_ids_illegal: list[str] = []
        answer_session_id_set = set(answer_session_ids)

        for session_path in session_files:
            try:
                session = self._load_json(session_path)
            except (ValueError, FileNotFoundError) as exc:
                self.logger.warning(f"[{self.name}] skip {session_path.name}: {exc}")
                continue

            session_id = str(session.get("haystack_session_id") or session_path.stem)
            session_date = str(session.get("haystack_date") or "").strip()
            session_dt = self._parse_lme_datetime(session_date)
            if session_dt is not None and session_dt > question_dt:
                session_ids_illegal.append(session_id)
                filtered_sessions.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "session_file": session_path.name,
                        "reason": "session_date_after_question_date",
                    },
                )
                if session_id in answer_session_id_set:
                    answer_session_ids_illegal.append(session_id)
                continue
            if session_dt is None:
                self.logger.warning(
                    f"[{self.name}] keep {session_id}: cannot parse haystack_date={session_date!r}",
                )
            sessions.append((session, session_id, session_date))

        illegal_answer_session_ids = set(answer_session_ids_illegal)
        answer_session_ids_filter_illegal = [
            session_id for session_id in answer_session_ids if session_id not in illegal_answer_session_ids
        ]
        total = len(sessions)
        self.logger.info(
            f"[{self.name}] reviewing {total} sessions from {session_dir} "
            f"(filtered {len(session_ids_illegal)} sessions after question_date, "
            f"start_interval={START_INTERVAL_SECONDS}s)",
        )

        failed_reviews: list[dict] = []

        async def wait_for_start_slot() -> None:
            await asyncio.to_thread(self._wait_for_global_start_slot)

        async def review_one(idx: int, session: dict, session_id: str, session_date: str) -> dict | None:
            user_prompt = self.prompt_format(
                "user_message",
                question=question,
                question_type=question_type,
                question_date=question_date,
                golden_answer=golden_answer,
                session_id=session_id,
                session_date=session_date,
                session_content=json.dumps(session.get("messages", []), ensure_ascii=False, indent=2),
            )
            try:
                await wait_for_start_slot()
                result = await self.agent_wrapper.reply(
                    user_prompt,
                    system_prompt=self.get_prompt("system_prompt"),
                    output_schema=_SESSION_REVIEW_SCHEMA,
                )
            except Exception as exc:  # noqa: BLE001 — one bad session must not abort the sweep
                self.logger.warning(f"[{self.name}] review failed for {session_id}: {exc}")
                failed_reviews.append(
                    {
                        "session_id": session_id,
                        "session_date": session_date,
                        "error": str(exc),
                    },
                )
                return None

            extracted = result.get("structured_output")
            if not isinstance(extracted, dict):
                # Fall back to the free-text reply when structured output is unavailable.
                extracted = {"relevant_info": (result.get("result") or "").strip()}
            relevant_info = str(extracted.get("relevant_info") or "").strip()

            summary = {
                "session_id": session_id,
                "session_date": session_date,
                "is_relevant": bool(extracted.get("is_relevant")) or bool(relevant_info),
                "relevant_info": relevant_info,
            }
            self.logger.info(
                f"[{self.name}] ({idx}/{total}) {session_id} " f"relevant={summary.get('is_relevant')}",
            )
            return summary

        # gather preserves input order, so summaries stay chronological.
        results = await asyncio.gather(
            *(
                review_one(idx, session, session_id, session_date)
                for idx, (session, session_id, session_date) in enumerate(sessions, start=1)
            ),
        )
        summaries: list[dict] = [s for s in results if s is not None]
        relevant_summaries = [s for s in summaries if s.get("is_relevant")]
        reviewed_session_ids = [str(s.get("session_id")) for s in summaries if s.get("session_id")]
        output = {
            "query": {
                "question_id": query_data.get("question_id"),
                "question": question,
                "question_type": question_type,
                "question_date": question_date,
            },
            "golden": {
                "answer": golden_answer,
                "answer_session_ids": answer_session_ids,
                "answer_session_ids_filter_illegal": answer_session_ids_filter_illegal,
                "answer_session_ids_illegal": answer_session_ids_illegal,
            },
            "review": {
                "num_session_files": len(session_files),
                "num_reviewed_sessions": len(summaries),
                "num_relevant_sessions": len(relevant_summaries),
                "num_irrelevant_sessions": len(summaries) - len(relevant_summaries),
                "num_failed_reviews": len(failed_reviews),
                "num_filtered_sessions": len(session_ids_illegal),
                "reviewed_session_ids": reviewed_session_ids,
                "session_ids_illegal": session_ids_illegal,
                "filtered_sessions": filtered_sessions,
                "failed_reviews": failed_reviews,
            },
            "session_summaries": relevant_summaries,
        }
        output_path = self.workspace_path / OUTPUT_FILENAME
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[{self.name}] wrote session review to {output_path}")

        self.context.response.success = True
        self.context.response.answer = f"reviewed {len(summaries)} sessions"
        self.context.response.metadata.update(
            {
                "num_session_files": len(session_files),
                "num_reviewed_sessions": len(summaries),
                "num_failed_reviews": len(failed_reviews),
                "num_filtered_sessions": len(session_ids_illegal),
                "output_path": str(output_path),
            },
        )
        return self.context.response
