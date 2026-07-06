"""Judge whether an agent answer matches the golden answer."""

import json
from typing import Any

from pydantic import BaseModel, Field

from ..base_step import BaseStep
from ...components import R


class AnswerJudgement(BaseModel):
    """Structured output for answer correctness judging."""

    thinking: str = Field(
        description="Brief reasoning comparing agent_answer to golden_answer for the query.",
    )
    answer: bool = Field(
        description="True if agent_answer correctly answers the query according to golden_answer; otherwise false.",
    )


@R.register("answer_judge_step")
class AnswerJudgeStep(BaseStep):
    """Evaluate whether an agent answer is correct against a golden answer."""

    SYS_PROMPT = """You judge whether agent_answer correctly answers query according to golden_answer.

Rules:
- Treat golden_answer as the source of truth.
- Mark correct only when agent_answer is semantically equivalent to golden_answer for the query.
- Minor wording differences are acceptable.
- Missing key facts, contradictions, unsupported extra claims that change the answer, or refusal when golden_answer answers should be marked false.
- The answer field must be a boolean.
- The thinking should briefly explain the comparison.
"""

    @staticmethod
    def _normalize_output(value: Any) -> dict[str, str | bool]:
        if isinstance(value, BaseModel):
            raw = value.model_dump()
        elif isinstance(value, dict):
            raw = value
        else:
            raw = {}

        result: dict[str, str | bool] = {}
        thinking = raw.get("thinking")
        if isinstance(thinking, str) and thinking.strip():
            result["thinking"] = thinking.strip()

        answer = raw.get("answer")
        if isinstance(answer, bool):
            result["answer"] = answer
        elif isinstance(answer, str):
            normalized = answer.strip().lower()
            if normalized in {"true", "yes", "correct", "对", "正确"}:
                result["answer"] = True
            elif normalized in {"false", "no", "incorrect", "错", "错误", "不正确"}:
                result["answer"] = False
        return result

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        agent_answer: str = self.context.get("agent_answer", "")
        golden_answer: str = self.context.get("golden_answer", "")

        if not query:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty query"
            return self.context.response
        if not agent_answer:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty agent_answer"
            return self.context.response
        if not golden_answer:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty golden_answer"
            return self.context.response
        if self.agent_wrapper is None:
            self.context.response.success = False
            self.context.response.answer = "Skipped: agent_wrapper is not configured"
            return self.context.response

        user_prompt = (
            f"query: {query}\n\n"
            f"agent_answer:\n{agent_answer}\n\n"
            f"golden_answer:\n{golden_answer}\n\n"
            "Judge whether agent_answer is correct."
        )
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.SYS_PROMPT,
            output_schema=AnswerJudgement,
        )

        raw_output = result.get("structured_output")
        output = self._normalize_output(raw_output)
        if "answer" not in output:
            output["answer"] = False
        answer = json.dumps(output, ensure_ascii=False, separators=(",", ":"))

        self.logger.info(f"[{self.name}] answer judgement: {answer}")
        self.context["answer_judgement"] = output
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query": query,
                "agent_answer": agent_answer,
                "golden_answer": golden_answer,
                "answer_judgement": output,
                "structured_output": raw_output,
            },
        )
        return self.context.response
