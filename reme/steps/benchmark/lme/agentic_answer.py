"""agentic_answer — answer the LongMemEval question from the indexed memory.

Job #4 of the pipeline. Reads ``query.json`` and hands the question to an agent
equipped with ``vector_search`` / ``bm25_search`` / ``python_execute`` /
``extract_session_by_id``. The agent searches the daily-note index, pivots to
raw sessions by ``session_id`` when a hit is promising, and keeps trying until it
can answer or has searched too many times. The final answer is written to
``mem_answer.json`` in the workspace.
"""

import json

from ...base_step import BaseStep
from ....components import R


@R.register("lme_agentic_answer_step")
class LmeAgenticAnswerStep(BaseStep):
    """Drive the tool-using agent that answers from indexed memory."""

    _OUTPUT_FILE = "mem_answer.json"

    def _load_query(self) -> dict:
        path = self.workspace_path / "query.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("query.json is not a JSON object")
        return data

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_agentic_answer_step requires agent_wrapper")

        query = self._load_query()
        question = str(query.get("question", "") or "").strip()
        question_date = str(query.get("question_date", "") or "").strip()
        question_id = str(query.get("question_id", "") or "").strip()
        if not question:
            raise ValueError("query.json requires a non-empty 'question'")

        user_prompt = self.prompt_format(
            "user_message",
            question=question,
            question_date=question_date or "(unknown)",
        )
        # A stable tool_context_id makes vector/bm25 dedup across this answer run,
        # so repeated searches surface genuinely new chunks each time.
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.get_prompt("system_prompt"),
            tool_context_id=question_id or question,
        )
        answer = (result.get("result") or "").strip()
        # session_id names the trajectory file mem_session/agentscope/<session_id>.jsonl,
        # so downstream tooling can locate this run's full tool-call trail.
        session_id = str(result.get("session_id") or "")

        out_path = self.workspace_path / self._OUTPUT_FILE
        out_path.write_text(
            json.dumps(
                {
                    "question_id": question_id,
                    "question": question,
                    "answer": answer,
                    "session_id": session_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self.logger.info(f"[{self.name}] answer for {question_id or question!r}: {answer!r}")
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
                "session_id": session_id,
                "path": self._OUTPUT_FILE,
            },
        )
        return self.context.response
