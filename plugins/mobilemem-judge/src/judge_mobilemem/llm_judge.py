"""Official MobileMem binary judge as an isolated ReMe Job."""

from dataclasses import asdict

from agentscope.message import UserMsg

from reme.steps.base_step import BaseStep
from reme.steps.evolve.compressor import CompressorStep

from .protocol import judge_prompt, parse_judge, protocol_id


class MobileMemJudgeStep(BaseStep):
    """Score saved predictions without accessing the candidate memory or tools."""

    async def execute(self):
        """Apply the unchanged benchmark prompt and verdict parser."""
        mode = self.context.get("judge_protocol", "membase_default")
        prompt = judge_prompt(
            self.context["question"],
            self.context["reference"],
            self.context["prediction"],
            self.context.get("evidence", ()),
            judge_protocol=mode,
        )
        response = await self.as_llm([UserMsg(name="user", content=prompt)])
        if hasattr(type(response), "__aiter__"):
            last = None
            async for chunk in response:
                last = chunk
            response = last
        raw = await CompressorStep._response_text(response)  # pylint: disable=protected-access
        parsed = parse_judge(raw)
        self.context.response.success = True
        self.context.response.answer = str(parsed["llm_judge_score"])
        self.context.response.metadata.update(
            **parsed,
            default_judge_protocol=protocol_id(mode),
            prompt=prompt,
            raw_response=raw,
            usage=asdict(response.usage) if response is not None and response.usage is not None else None,
        )
        return self.context.response
