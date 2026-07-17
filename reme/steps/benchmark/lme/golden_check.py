"""Judge whether the LongMemEval golden answer is reasonable.

Consumes ``session_review.json`` produced by ``lme_session_review_step`` and
hands its extracted session information to an agent that is equipped with the
``python_execute`` tool. The agent uses ``python_execute`` only as a scratchpad
for checking the golden answer; ``answer_session_ids`` are outside the audit
scope. The final verdict is not the
raw Python stdout but a *structured* object extracted from the whole conversation
via ``output_schema``. Sessions dated after ``question_date`` are filtered
upstream by ``lme_session_review_step`` and are not included in this
golden-check flow.

The output ``check_golden.json`` is intentionally slim: it does NOT duplicate the
query/golden/review fields already stored in ``session_review.json`` (referenced by
path), keeping only the relevant per-session ``session_summaries`` and the
structured verdict. It is written to the workspace root (e.g.
``datasets/longmemeval/1/check_golden.json``).
"""

import json
import asyncio
from uuid import uuid4

from ...base_step import BaseStep
from ....components import R

# File written under the workspace root with the full review + verdict payload.
OUTPUT_FILENAME = "check_golden.json"
SESSION_REVIEW_FILENAME = "session_review.json"
RETRY_INITIAL_SECONDS = 5.0
RETRY_MAX_SECONDS = 300.0

# Structured verdict the judge agent must produce (extracted from its reasoning).
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "用中文写出详细的推理过程：先说明证据支持的答案，再判断 " "golden_answer 是否正确。",
        },
        "golden_answer_correct": {
            "type": "boolean",
            "description": "golden_answer 是否正确。",
        },
        "true_answer": {
            "type": "string",
            "description": "仅当 golden_answer_correct 为 false 时填写：证据支持的正确答案"
            "（证据不足时填 'unknown'）。golden_answer_correct 为 true 时填空字符串。",
        },
    },
    "required": [
        "reasoning",
        "golden_answer_correct",
        "true_answer",
    ],
    "additionalProperties": False,
}


@R.register("lme_golden_check_step")
class GoldenCheckStep(BaseStep):
    """Let a python-enabled agent decide whether the golden answer holds up."""

    @staticmethod
    def _compact_summary(summary: dict) -> dict:
        """Keep only the evidence fields used by the golden-check prompt."""
        return {
            "session_id": str(summary.get("session_id") or ""),
            "session_date": str(summary.get("session_date") or ""),
            "extracted_info": str(summary.get("extracted_info") or ""),
        }

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_golden_check_step requires agent_wrapper")

        review_path = self.workspace_path / SESSION_REVIEW_FILENAME
        if not review_path.is_file():
            raise FileNotFoundError(
                f"{SESSION_REVIEW_FILENAME} not found at {review_path}; run lme_session_review_step first",
            )
        try:
            with review_path.open(encoding="utf-8") as f:
                review_payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {review_path}") from exc
        if not isinstance(review_payload, dict):
            raise ValueError(f"Expected a JSON object in {review_path}")

        query = review_payload.get("query") or {}
        golden = review_payload.get("golden") or {}
        # session_review.json keeps one extraction per reviewed session.
        session_summaries = [self._compact_summary(s) for s in review_payload.get("session_summaries") or []]

        question = str(query.get("question") or "").strip()
        question_type = str(query.get("question_type") or "").strip()
        question_date = str(query.get("question_date") or "").strip()
        golden_answer = str(golden.get("answer") or "").strip()
        if not question:
            raise ValueError(f"{review_path} does not contain a question")

        prompt_input = {
            "question": question,
            "question_type": question_type,
            "question_date": question_date,
            "golden_answer": golden_answer,
            "session_summaries": session_summaries,
        }
        user_prompt = self.prompt_format(
            "user_message",
            question=question,
            question_type=question_type,
            question_date=question_date,
            golden_answer=golden_answer,
            num_session_summaries=len(session_summaries),
            payload_json=json.dumps(prompt_input, ensure_ascii=False, indent=2),
        )

        tool_context_id = str(self.context.get("tool_context_id") or f"lme-golden-{uuid4()}")
        retry_initial_seconds = float(self.kwargs.get("retry_initial_seconds", RETRY_INITIAL_SECONDS))
        retry_max_seconds = float(self.kwargs.get("retry_max_seconds", RETRY_MAX_SECONDS))
        retry_max_attempts_raw = self.kwargs.get("retry_max_attempts")
        retry_max_attempts = int(retry_max_attempts_raw) if retry_max_attempts_raw not in (None, "") else 0
        if retry_initial_seconds <= 0:
            retry_initial_seconds = RETRY_INITIAL_SECONDS
        retry_max_seconds = max(retry_max_seconds, retry_initial_seconds)

        attempt = 1
        sleep_seconds = retry_initial_seconds
        while True:
            try:
                result = await self.agent_wrapper.reply(
                    user_prompt,
                    system_prompt=self.get_prompt("system_prompt"),
                    tool_context_id=tool_context_id,
                    output_schema=_VERDICT_SCHEMA,
                )
                if attempt > 1:
                    self.logger.info(f"[{self.name}] golden check recovered after {attempt} attempts")
                break
            except Exception as exc:
                if 0 < retry_max_attempts <= attempt:
                    raise
                next_sleep = min(sleep_seconds, retry_max_seconds)
                self.logger.warning(
                    f"[{self.name}] golden check attempt {attempt} failed: {exc}; " f"retrying in {next_sleep:.1f}s",
                )
                await asyncio.sleep(next_sleep)
                sleep_seconds = min(sleep_seconds * 2, retry_max_seconds)
                attempt += 1

        # The structured verdict is the real output; the free-text reply is only the
        # agent's closing narration and is kept as a fallback.
        verdict = result.get("structured_output")
        if not isinstance(verdict, dict):
            self.logger.warning(f"[{self.name}] no structured verdict; falling back to free text")
            verdict = {"reasoning": (result.get("result") or "").strip()}
        # Retain the legacy fields for readers of existing check_golden.json
        # artifacts. They are compatibility placeholders, not audit results.
        verdict["answer_session_ids_correct"] = True
        verdict["true_answer_session_ids"] = []

        # Slim output: do NOT duplicate session_review.json (referenced by path);
        # keep only the compact session_summaries and the verdict.
        output = {
            "session_review_path": str(review_path),
            "session_summaries": session_summaries,
            "verdict": verdict,
        }
        output_path = self.workspace_path / OUTPUT_FILENAME
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[{self.name}] wrote verdict to {output_path}")

        self.context.response.success = True
        self.context.response.answer = json.dumps(verdict, ensure_ascii=False, indent=2)
        self.context.response.metadata.update(
            {
                "num_session_summaries": len(session_summaries),
                "session_review_path": str(review_path),
                "tool_context_id": tool_context_id,
                "agent_session_id": result.get("session_id"),
                "output_path": str(output_path),
                "verdict": verdict,
            },
        )
        return self.context.response
