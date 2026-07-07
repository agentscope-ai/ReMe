"""Agentic LongMemEval answer step backed by AgentScope search tools."""

import json
from uuid import uuid4

from ....components import R
from ...base_step import BaseStep


@R.register("agentic_answer_step")
class AgenticAnswerStep(BaseStep):
    """Answer a LongMemEval query by letting an agent search indexed history."""

    def _load_query(self) -> dict:
        path = self.workspace_path / "query.json"
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            raise FileNotFoundError(f"Cannot read LongMemEval query file: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid LongMemEval query JSON: {path}") from exc

        question = str(data.get("question") or "").strip()
        if not question:
            raise ValueError(f"LongMemEval query file requires non-empty question: {path}")
        return data

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("agentic_answer_step requires agent_wrapper")

        query_data = self._load_query()
        question = str(query_data.get("question") or "").strip()
        question_date = str(query_data.get("question_date") or "").strip()
        question_id = str(query_data.get("question_id") or "").strip()
        tool_context_id = str(self.context.get("tool_context_id") or f"agentic-search-{uuid4()}").strip()

        user_prompt = self.prompt_format(
            "user_message",
            question=question,
            question_date=question_date,
        )
        result = await self.agent_wrapper.reply(
            user_prompt,
            tool_context_id=tool_context_id,
            tool_result_offload_message=(
                "<system-reminder>"
                "The tool result is too long. Its core content has been saved "
                "to the draft. If you need to read the core content from "
                "historical tool calls, use read_all_draft to read all saved "
                "memory results."
                "</system-reminder>"
            ),
        )
        answer = (result.get("result") or "").strip()

        self.logger.info(f"[{self.name}] agentic search answer: {answer}")
        self.context["query"] = question
        self.context["question_date"] = question_date
        self.context["context_answer"] = answer
        self.context["agent_answer"] = answer
        self.context["tool_context_id"] = tool_context_id
        self.context["agent_session_id"] = result.get("session_id")
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query": question,
                "question_date": question_date,
                "question_id": question_id,
                "context_answer": answer,
                "tool_context_id": tool_context_id,
                "agent_session_id": result.get("session_id"),
            },
        )
        return self.context.response
