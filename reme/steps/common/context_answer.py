"""Answer a query strictly from the supplied session context."""

import json
from typing import Any

from pydantic import BaseModel, Field

from ..base_step import BaseStep
from ...components import R


class ContextAnswer(BaseModel):
    """Structured output for context-grounded answering."""

    thinking: str = Field(
        description=(
            "Brief reasoning that identifies the exact context support for the answer. "
            "If the context is insufficient, briefly explain what key evidence is missing."
        ),
    )
    answer: str = Field(
        description=(
            "Required non-empty final answer to the user query, strictly supported by "
            'the provided context. Always answer this field. If the context is '
            'insufficient or does not contain the answer, set this field exactly to "unknown".'
        ),
    )


@R.register("context_answer_step")
class ContextAnswerStep(BaseStep):
    """Answer a query using only the provided session context."""

    SYS_PROMPT = """You answer the user query strictly from the provided context.

Rules:
- Use only facts explicitly supported by context.
- Do not use outside knowledge.
- Do not guess, infer unstated details, or fill gaps.
- The answer field is required and must be non-empty.
- If context is insufficient to answer, still answer by setting answer exactly to "unknown".
- The thinking should briefly cite what in the context supports the answer, or why it does not.
"""

    @staticmethod
    def _normalize_output(value: Any) -> dict[str, str]:
        if isinstance(value, BaseModel):
            raw = value.model_dump()
        elif isinstance(value, dict):
            raw = value
        else:
            raw = {}

        result: dict[str, str] = {}
        for key in ("thinking", "answer"):
            item = raw.get(key)
            if isinstance(item, str):
                item = item.strip()
                if item:
                    result[key] = item
        return result

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        session_context: str = self.context.get("session_context", "")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response
        if not session_context:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty session_context"
            return self.context.response
        if self.agent_wrapper is None:
            self.context.response.success = False
            self.context.response.answer = "Skipped: agent_wrapper is not configured"
            return self.context.response

        user_prompt = f"query: {query}\n\nsession_context:\n{session_context}\n\nAnswer the query strictly from context."
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.SYS_PROMPT,
            output_schema=ContextAnswer,
        )

        raw_output = result.get("structured_output")
        output = self._normalize_output(raw_output)
        if "answer" not in output:
            output["answer"] = "unknown"
        answer = json.dumps(output, ensure_ascii=False, separators=(",", ":"))

        self.logger.info(f"[{self.name}] context answer: {answer}")
        self.context["context_answer"] = output
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query": query,
                "session_context": session_context,
                "context_answer": output,
                "structured_output": raw_output,
            },
        )
        return self.context.response
