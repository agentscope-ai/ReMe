"""Judge whether the LongMemEval golden answer is reasonable.

Consumes the per-session summaries produced by ``lme_session_review_step`` along
with the query and answer, and hands them to an agent that is equipped with the
``python_execute`` tool. The agent uses ``python_execute`` only as a scratchpad
for the hard reasoning (counting relevant sessions, cross-checking
``answer_session_ids``, comparing dates); the final verdict is not the raw Python
stdout but a *structured* object extracted from the whole conversation via
``output_schema``.

Finally the step assembles one big JSON — the original query/answer fields (kept
verbatim, never round-tripped through the LLM), every per-session structured
summary, and the structured verdict — and writes it to ``check_golden.json`` at
the workspace root (e.g. ``datasets/longmemeval/1/check_golden.json``).
"""

import json
from uuid import uuid4

from ...base_step import BaseStep
from ....components import R

# File written under the workspace root with the full review + verdict payload.
OUTPUT_FILENAME = "check_golden.json"

# Structured verdict the judge agent must produce (extracted from its reasoning).
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "golden_answer_reasonable": {
            "type": "boolean",
            "description": "Whether the golden answer is reasonable given the evidence.",
        },
        "true_answer": {
            "type": "string",
            "description": "When golden_answer_reasonable is true, echo the golden answer; when false, "
            "state the true answer the evidence supports (or 'unknown' if the evidence is insufficient).",
        },
        "answer_session_ids_reasonable": {
            "type": "boolean",
            "description": "Whether the question can be answered from exactly the given answer_session_ids.",
        },
        "true_answer_session_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "When answer_session_ids_reasonable is true, echo the given ids; when false, list "
            "the session ids that truly support the answer (empty list if none can).",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this verdict, between 0 and 1.",
        },
        "answer_session_date_ok": {
            "type": "boolean",
            "description": "Whether every answer session's date is not later than the question date.",
        },
        "relevant_session_count": {
            "type": "integer",
            "description": "How many reviewed sessions were flagged relevant.",
        },
        "supporting_session_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Session ids that actually support the answer.",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any problems found (empty list if none).",
        },
        "reasoning": {
            "type": "string",
            "description": "Short explanation of the verdict.",
        },
    },
    "required": [
        "golden_answer_reasonable",
        "true_answer",
        "answer_session_ids_reasonable",
        "true_answer_session_ids",
        "confidence",
        "answer_session_date_ok",
        "relevant_session_count",
        "supporting_session_ids",
        "issues",
        "reasoning",
    ],
}


@R.register("lme_golden_check_step")
class GoldenCheckStep(BaseStep):
    """Let a python-enabled agent decide whether the golden answer holds up."""

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_golden_check_step requires agent_wrapper")

        query_data = self.context.get("query_data") or {}
        answer_data = self.context.get("answer_data") or {}
        session_summaries = self.context.get("session_summaries") or []
        question = str(self.context.get("question") or query_data.get("question") or "").strip()
        question_type = str(self.context.get("question_type") or query_data.get("question_type") or "").strip()
        question_date = str(self.context.get("question_date") or query_data.get("question_date") or "").strip()
        golden_answer = str(self.context.get("golden_answer") or answer_data.get("answer") or "").strip()
        answer_session_ids = self.context.get("answer_session_ids") or answer_data.get("answer_session_ids") or []

        if not question:
            raise ValueError("lme_golden_check_step requires a question (run lme_session_review_step first)")
        if not session_summaries:
            raise ValueError("lme_golden_check_step requires session_summaries (run lme_session_review_step first)")

        payload = {
            "question": question,
            "question_type": question_type,
            "question_date": question_date,
            "golden_answer": golden_answer,
            "answer_session_ids": answer_session_ids,
            "session_summaries": session_summaries,
        }

        user_prompt = self.prompt_format(
            "user_message",
            question=question,
            question_type=question_type,
            question_date=question_date,
            golden_answer=golden_answer,
            answer_session_ids=", ".join(str(s) for s in answer_session_ids) or "(none)",
            num_sessions=len(session_summaries),
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
        )

        tool_context_id = str(self.context.get("tool_context_id") or f"lme-golden-{uuid4()}")
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.get_prompt("system_prompt"),
            tool_context_id=tool_context_id,
            output_schema=_VERDICT_SCHEMA,
        )

        # The structured verdict is the real output; the free-text reply is only the
        # agent's closing narration and is kept for debugging.
        verdict = result.get("structured_output")
        narration = (result.get("result") or "").strip()
        if not isinstance(verdict, dict):
            self.logger.warning(f"[{self.name}] no structured verdict; falling back to free text")
            verdict = {"reasoning": narration}

        # Assemble the big JSON: original (non-LLM) fields + per-session structured
        # summaries + structured verdict, then persist it at the workspace root.
        output = {
            "question_id": query_data.get("question_id"),
            "question": question,
            "question_type": question_type,
            "question_date": question_date,
            "golden_answer": golden_answer,
            "answer_session_ids": answer_session_ids,
            "num_sessions": len(session_summaries),
            "session_summaries": session_summaries,
            "verdict": verdict,
        }
        output_path = self.workspace_path / OUTPUT_FILENAME
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"[{self.name}] wrote review + verdict to {output_path}")

        self.context["golden_check"] = verdict
        self.context["golden_check_path"] = str(output_path)
        self.context.response.success = True
        self.context.response.answer = json.dumps(verdict, ensure_ascii=False, indent=2)
        self.context.response.metadata.update(
            {
                "question": question,
                "question_date": question_date,
                "golden_answer": golden_answer,
                "answer_session_ids": answer_session_ids,
                "num_sessions": len(session_summaries),
                "tool_context_id": tool_context_id,
                "agent_session_id": result.get("session_id"),
                "output_path": str(output_path),
                "verdict": verdict,
            },
        )
        return self.context.response
