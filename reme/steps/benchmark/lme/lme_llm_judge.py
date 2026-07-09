"""LME file-based LLM judge step."""

import json
import re

from ...base_step import BaseStep
from ....components import R


@R.register("lme_llm_judge_step")
class LmeLlmJudgeStep(BaseStep):
    """Judge ``mem_answer.json`` against ``answer.json`` and update it in place."""

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
    def _user_prompt_key(judge_prompt_key: str) -> str:
        if judge_prompt_key == "single_session_preference_system_prompt":
            return "preference_judge_user_message"
        return "answer_judge_user_message"

    @staticmethod
    def _normalize_judgement(raw_answer: str) -> str:
        match = re.match(r"\s*(yes|no)\b", raw_answer, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return raw_answer.strip().lower()

    def _load_json_object(self, filename: str) -> dict:
        path = self.workspace_path / filename
        if not path.exists():
            raise FileNotFoundError(f"{filename} does not exist in {self.workspace_path}")
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{filename} is not a JSON object")
        return data

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise RuntimeError("lme_llm_judge_step requires agent_wrapper")

        query_data = self._load_json_object("query.json")
        golden_data = self._load_json_object("answer.json")
        mem_answer = self._load_json_object("mem_answer.json")

        query = str(query_data.get("question", "") or "").strip()
        agent_answer = str(mem_answer.get("answer", "") or "").strip()
        golden_answer = str(golden_data.get("answer", "") or "").strip()
        question_type = str(query_data.get("question_type", "") or "")

        if not query:
            raise ValueError("query.json requires a non-empty 'question'")
        if not agent_answer:
            raise ValueError("mem_answer.json requires a non-empty 'answer'")
        if not golden_answer:
            raise ValueError("answer.json requires a non-empty 'answer'")

        judge_prompt_key = self._judge_prompt_key(question_type)
        user_prompt = self.prompt_format(
            self._user_prompt_key(judge_prompt_key),
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

        mem_answer["llm_judge"] = {
            "judgement": answer,
            "raw_judgement": raw_answer,
            "golden_answer": golden_answer,
            "question_type": question_type,
        }
        out_path = self.workspace_path / "mem_answer.json"
        out_path.write_text(json.dumps(mem_answer, ensure_ascii=False, indent=2), encoding="utf-8")

        question_id = str(query_data.get("question_id") or mem_answer.get("question_id") or "")
        self.logger.info(f"[{self.name}] llm judgement for {question_id or query!r}: {answer}")
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "question_id": question_id,
                "query": query,
                "agent_answer": agent_answer,
                "golden_answer": golden_answer,
                "question_type": question_type,
                "answer_judgement": answer,
                "raw_answer_judgement": raw_answer,
                "path": "mem_answer.json",
            },
        )
        return self.context.response
