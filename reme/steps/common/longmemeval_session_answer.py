"""Answer a LongMemEval query by searching only its sibling session directory."""

import json
from pathlib import Path

from ..base_step import BaseStep
from ...components import R


@R.register("longmemeval_session_answer_step")
class LongMemEvalSessionAnswerStep(BaseStep):
    """Ask an agent to answer a LongMemEval query from the sibling session files only."""

    SYS_PROMPT = """You answer a LongMemEval user question by searching session files.

Strict source rules:
- You may read the provided query_json_path.
- You may read files under the provided session_dir.
- Do not read, list, grep, or infer from any other file or directory.
- Do not use web search or outside knowledge.

Search rules:
- Do not give up after one failed search.
- Search in multiple rounds with different keywords, names, dates, preferences,
  entities, and paraphrases from the question.
- Inspect promising session files before answering.
- Answer strictly according to evidence found in session_dir.
- If the session files do not contain enough evidence, answer exactly: unknown

Final response:
- Return only the final answer text for the user question.
- Do not include citations, file paths, analysis, or markdown.
"""

    @staticmethod
    def _resolve_paths(raw_query_path: str) -> tuple[Path | None, Path | None, str | None]:
        if not raw_query_path:
            return None, None, "Skipped: empty query_path"

        query_path = Path(raw_query_path).expanduser()
        if not query_path.is_absolute():
            query_path = Path.cwd() / query_path
        try:
            query_path = query_path.resolve()
            longmemeval_root = (Path.cwd() / "longmemeval").resolve()
        except OSError as exc:
            return None, None, f"Invalid query_path: {exc}"

        if query_path.name != "query.json":
            return None, None, "Invalid query_path: expected a query.json file"
        if longmemeval_root != query_path and longmemeval_root not in query_path.parents:
            return None, None, "Invalid query_path: must be under longmemeval"
        if not query_path.is_file():
            return None, None, f"query.json not found: {query_path}"

        session_dir = query_path.parent / "session"
        if not session_dir.is_dir():
            return None, None, f"session directory not found: {session_dir}"
        return query_path, session_dir.resolve(), None

    async def execute(self):
        assert self.context is not None
        raw_query_path: str = self.context.get("query_path") or self.context.get("query_file") or ""
        self.logger.info(f"[{self.name}] longmemeval input query_path={raw_query_path!r}")

        query_path, session_dir, error = self._resolve_paths(raw_query_path)
        if error is not None:
            self.logger.warning(f"[{self.name}] longmemeval invalid input: {error}")
            self.context.response.success = False
            self.context.response.answer = error
            return self.context.response
        if self.agent_wrapper is None:
            self.context.response.success = False
            self.context.response.answer = "Skipped: agent_wrapper is not configured"
            return self.context.response
        assert query_path is not None
        assert session_dir is not None
        try:
            query_payload = json.loads(query_path.read_text(encoding="utf-8"))
            question = query_payload.get("question") if isinstance(query_payload, dict) else None
        except (OSError, json.JSONDecodeError):
            question = None
        session_count = sum(1 for path in session_dir.iterdir() if path.is_file())
        self.logger.info(
            f"[{self.name}] longmemeval resolved query_path={query_path} "
            f"session_dir={session_dir} session_files={session_count} question={question!r}",
        )

        prompt = (
            f"query_json_path: {query_path}\n"
            f"session_dir: {session_dir}\n\n"
            "Read the query JSON, then search only session_dir for the answer. "
            "Use multiple search rounds if needed. Return only the final answer."
        )
        self.logger.info(f"[{self.name}] longmemeval calling agent_wrapper")
        result = await self.agent_wrapper.reply(prompt, system_prompt=self.SYS_PROMPT)
        answer = (result.get("result") or "").strip() or "unknown"

        self.logger.info(f"[{self.name}] longmemeval session answer: {answer!r}")
        self.context["longmemeval_session_answer"] = answer
        self.context.response.success = True
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "query_path": str(query_path),
                "session_dir": str(session_dir),
                "answer": answer,
            },
        )
        return self.context.response
