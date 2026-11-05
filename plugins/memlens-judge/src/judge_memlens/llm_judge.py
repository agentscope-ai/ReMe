"""Official MemLens task-specific judge as an isolated ReMe Job."""

from dataclasses import asdict

from agentscope.message import UserMsg

from reme.steps.base_step import BaseStep
from reme.steps.evolve.compressor import CompressorStep

from .protocol import (
    SOURCE_REVISION,
    judge_prompt,
    normalize_for_judge,
    parse_judge_response,
    task_key,
)


class MemLensJudgeStep(BaseStep):
    """Score saved answers without accessing the candidate's workspace or tools."""

    async def execute(self):
        """Apply the official judge prompt/parser to one saved prediction."""
        question = self.context["question"]
        reference = self.context["reference"]
        prediction = self.context["prediction"]
        category = self.context["category"]
        subtype = self.context.get("subtype", "")
        normalized = normalize_for_judge(prediction)
        prompt = judge_prompt(
            question,
            reference,
            normalized["text"],
            category,
            subtype,
            self.context.get("old_answer", ""),
        )
        words = len(normalized["text"].split())
        result = {
            "source_revision": SOURCE_REVISION,
            "task_key": task_key(category, subtype, reference),
            "original_len": normalized["original_len"],
            "was_truncated": normalized["was_truncated"],
            "is_degenerate": normalized["is_degenerate"],
            **normalized["diagnostics"],
            "parsed_words": words,
            "auto_zero": words > 500,
        }
        raw, usage = "", None
        if words > 500:
            parsed = {"acc": 0.0, "parse_default_zero": False}
        else:
            response = await self.as_llm([UserMsg(name="user", content=prompt)])
            if hasattr(type(response), "__aiter__"):
                last = None
                async for chunk in response:
                    last = chunk
                response = last
            raw = await CompressorStep._response_text(response)  # pylint: disable=protected-access
            usage = asdict(response.usage) if response is not None and response.usage is not None else None
            parsed = parse_judge_response(raw)
        score = parsed["acc"]
        self.context.response.success = True
        self.context.response.answer = str(score)
        self.context.response.metadata.update(
            **result,
            **parsed,
            llm_judge_score=score,
            raw_response=raw,
            usage=usage,
        )
        return self.context.response
