"""LongMemEval judge preparation and result persistence steps."""

import json
from pathlib import Path
from typing import Any

from ....components import R
from ...base_step import BaseStep


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read LongMemEval {label} file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LongMemEval {label} JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"LongMemEval {label} file must be a JSON object: {path}")
    return data


@R.register("lme_prepare_judge_step")
class LmePrepareJudgeStep(BaseStep):
    """Load LongMemEval golden fields and normalize context for answer_judge_step."""

    async def execute(self):
        assert self.context is not None

        query_data = _read_json(self.workspace_path / "query.json", "query")
        answer_data = _read_json(self.workspace_path / "answer.json", "answer")

        query = str(query_data.get("question") or self.context.get("query") or "").strip()
        agent_answer = str(self.context.get("context_answer") or self.context.get("agent_answer") or "").strip()
        golden_answer = str(answer_data.get("answer") or "").strip()
        question_type = str(query_data.get("question_type") or "").strip()

        if not query:
            raise ValueError("lme_prepare_judge_step requires non-empty question")
        if not agent_answer:
            raise ValueError("lme_prepare_judge_step requires non-empty agent answer")
        if not golden_answer:
            raise ValueError("lme_prepare_judge_step requires non-empty golden answer")

        prepared = {
            "query": query,
            "question_id": str(query_data.get("question_id") or "").strip(),
            "question_type": question_type,
            "question_date": str(query_data.get("question_date") or "").strip(),
            "agent_answer": agent_answer,
            "golden_answer": golden_answer,
            "answer_session_ids": answer_data.get("answer_session_ids", []),
        }
        self.context.update(prepared)
        self.context.response.success = True
        self.context.response.answer = agent_answer
        self.context.response.metadata.update(prepared)
        return self.context.response


@R.register("lme_save_result_step")
class LmeSaveResultStep(BaseStep):
    """Save one LongMemEval answer-and-judge result under workspace/result."""

    async def execute(self):
        assert self.context is not None

        answer_id = str(self.context.get("answer_id") or "").strip()
        if not answer_id:
            raise ValueError("lme_save_result_step requires answer_id")
        if Path(answer_id).name != answer_id:
            raise ValueError("lme_save_result_step answer_id must be a file name, not a path")

        result_dir = self.workspace_path / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{answer_id}.json"

        payload = {
            "answer_id": answer_id,
            "index": self.workspace_path.name,
            "question_id": self.context.get("question_id", ""),
            "question_type": self.context.get("question_type", ""),
            "question_date": self.context.get("question_date", ""),
            "query": self.context.get("query", ""),
            "agent_answer": self.context.get("agent_answer", ""),
            "golden_answer": self.context.get("golden_answer", ""),
            "answer_session_ids": self.context.get("answer_session_ids", []),
            "answer_judgement": self.context.get("answer_judgement", ""),
            "raw_answer_judgement": self.context.response.metadata.get("raw_answer_judgement", ""),
            "context_answer": self.context.get("context_answer", ""),
            "tool_context_id": self.context.get(
                "tool_context_id",
                self.context.response.metadata.get("tool_context_id", ""),
            ),
            "agent_session_id": self.context.response.metadata.get("agent_session_id", ""),
            "success": self.context.response.success,
        }

        with result_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        self.context.response.success = True
        self.context.response.answer = str(result_path)
        self.context.response.metadata.update({"result_path": str(result_path), **payload})
        return self.context.response
