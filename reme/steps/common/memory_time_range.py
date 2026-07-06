"""Extract a precise memory retrieval time range from a user question."""

import json
from typing import Any

from ..base_step import BaseStep
from ...components import R


@R.register("memory_time_range_step")
class MemoryTimeRangeStep(BaseStep):
    """Infer a precise query time range for memory retrieval.

    Inputs:
        question (str, required): user question.
        question_date (str, required): reference date for relative dates.

    Output:
        JSON object string containing start_dt/end_dt when precise, or {}.
    """

    SYS_PROMPT = """You extract broad-but-useful time filters for memory retrieval.

Return dates that are explicitly stated by the user or can be resolved relative
to question_date. Use ISO 8601 dates or datetimes.

Final output:
- Return exactly one JSON object and nothing else. Do not use markdown.
- Allowed keys: thinking, start_dt, end_dt.
- If no time range is needed, return {}.

Rules:
- If the question has no precise date/time constraint, return an empty object.
- Do not infer a time range from vague words like "recent", "lately", "before",
  "nowadays", or topic words that are not time constraints.
- Prefer a looser range or an empty range over an overly narrow range. The range
  should avoid excluding likely relevant memories.
- Resolve relative expressions such as today, yesterday, tomorrow, this week,
  last week, this month, last month, this year, last year, past N days, and next
  N days using question_date.
- If resolving start_dt or end_dt requires any date arithmetic, calendar
  boundary calculation, weekday calculation, month/year length handling, or
  relative date calculation, you must call the python_execute tool first.
- Python date calculations must import datetime and print() the final JSON
  object. Then your final answer must repeat that printed JSON object exactly.
- For a single day, set both start_dt and end_dt to that date.
- For open-ended precise constraints, set only start_dt or only end_dt.
- Preserve only fields that are needed: thinking, start_dt, end_dt.
"""

    @staticmethod
    def _normalize_range(value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            raw = value
        elif isinstance(value, str):
            try:
                raw = json.loads(value)
            except json.JSONDecodeError:
                raw = {}
        else:
            raw = {}

        result: dict[str, str] = {}
        for key in ("thinking", "start_dt", "end_dt"):
            item = raw.get(key)
            if isinstance(item, str):
                item = item.strip()
                if item and item.lower() not in {"null", "none", "n/a", "unknown"}:
                    result[key] = item
        return result

    async def execute(self):
        assert self.context is not None
        question: str = self.context.get("question") or self.context.get("query") or ""
        question_date: str = self.context.get("question_date", "")

        if not question:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty question"
            return self.context.response
        if not question_date:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty question_date"
            return self.context.response
        if self.agent_wrapper is None:
            self.context.response.success = False
            self.context.response.answer = "Skipped: agent_wrapper is not configured"
            return self.context.response

        user_prompt = (
            f"question_date: {question_date}\n"
            f"question: {question}\n\n"
            "Extract the memory retrieval time range."
        )
        result = await self.agent_wrapper.reply(
            user_prompt,
            system_prompt=self.SYS_PROMPT,
            job_tools=["python_execute"],
        )

        raw_range = result.get("result") or ""
        time_range = self._normalize_range(raw_range)
        answer = json.dumps(time_range, ensure_ascii=False, separators=(",", ":"))

        self.logger.info(f"[{self.name}] memory time range: {answer}")
        self.context["memory_time_range"] = time_range
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "question": question,
                "question_date": question_date,
                "memory_time_range": time_range,
                "raw_result": raw_range,
            },
        )
        return self.context.response
