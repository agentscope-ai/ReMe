"""Judge whether an agent answer matches the golden answer."""

import re

from ...base_step import BaseStep
from ....components import R


@R.register("lme_answer_judge_step")
class LmeAnswerJudgeStep(BaseStep):
    """Evaluate whether an agent answer is correct against a golden answer."""

    PROMPT_KEYS_BY_QUESTION_TYPE = {
        "temporal_reasoning": "temporal_reasoning_system_prompt",
        "knowledge_update": "knowledge_update_system_prompt",
        "single_session_preference": "single_session_preference_system_prompt",
    }

    @classmethod
    def _judge_prompt_key(cls, question_type: str) -> str:
        normalized = question_type.strip().lower().replace("-", "_").replace(" ", "_")
        return cls.PROMPT_KEYS_BY_QUESTION_TYPE.get(normalized, "other_question_types_system_prompt")

    @staticmethod
    def _normalize_judgement(raw_answer: str) -> str:
        match = re.match(r"\s*(yes|no)\b", raw_answer, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return raw_answer.strip().lower()

    async def execute(self):
        assert self.context is not None
        query: str = self.context.get("query", "")
        agent_answer: str = self.context.get("agent_answer", "")
        golden_answer: str = self.context.get("golden_answer", "")
        question_type: str = self.context.get("question_type", "")

        if not query:
            raise ValueError("lme_answer_judge_step requires non-empty query")
        if not agent_answer:
            raise ValueError("lme_answer_judge_step requires non-empty agent_answer")
        if not golden_answer:
            raise ValueError("lme_answer_judge_step requires non-empty golden_answer")
        if self.agent_wrapper is None:
            raise RuntimeError("lme_answer_judge_step requires agent_wrapper")

        judge_prompt_key = self._judge_prompt_key(question_type)
        user_prompt_key = (
            "preference_judge_user_message"
            if judge_prompt_key == "single_session_preference_system_prompt"
            else "answer_judge_user_message"
        )
        user_prompt = self.prompt_format(
            user_prompt_key,
            query=query,
            golden_answer=golden_answer,
            agent_answer=agent_answer,
        )
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.prompt_format(judge_prompt_key),
        )

        raw_answer = (result.get("result") or "").strip()
        answer = self._normalize_judgement(raw_answer)

        self.logger.info(f"[{self.name}] answer judgement: {answer}")
        self.context["answer_judgement"] = answer
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query": query,
                "agent_answer": agent_answer,
                "golden_answer": golden_answer,
                "question_type": question_type,
                "answer_judgement": answer,
                "raw_answer_judgement": raw_answer,
            },
        )
        return self.context.response
