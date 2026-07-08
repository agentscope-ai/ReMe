"""lme_auto_memory — turn every LongMemEval session into a search-friendly note.

For a workspace such as ``datasets/longmemeval/1`` this step walks each raw
session under ``resource_dir`` (files named ``<date>_(...)_<time>@<session_id>.json``
with ``haystack_date`` / ``haystack_session_id`` / ``messages``) and, one per
session, asks an agent to *completely* extract its content — entities, times,
numbers, preferences, events, causal links — into a daily note optimized for
both BM25 and vector retrieval.

Each note is written to ``<daily_dir>/<YYYY-MM-DD>/<name>.md`` via the shared
``daily_write`` job, so the frontmatter carries ``session_id`` for progressive
expansion (the agentic-answer flow pivots from a search hit back to the raw
session through this id). Filenames are LLM-generated topic stems; same-day
collisions are disambiguated by appending the session id.
"""

import asyncio
import json
from pathlib import Path

import frontmatter

from ...base_step import BaseStep
from ...file_io import extract_daily_date, validate_session_id
from ....components import R

DEFAULT_CONCURRENCY = 12

# Structured extraction the memory agent must return per session.
_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Concise, stable topic/event filename stem (kebab-case, no date, no slash or "
            "reserved characters). E.g. 'daily-commute-details' or 'leather-boot-care'.",
        },
        "description": {
            "type": "string",
            "description": "Thorough one-paragraph summary of the note body — specific enough that this "
            "description alone conveys all key facts. Used as a search-friendly abstract.",
        },
        "body": {
            "type": "string",
            "description": "Complete markdown extraction of every core fact in the session, written for "
            "retrieval (natural-language statements, explicit entities, dates and numbers verbatim).",
        },
    },
    "required": ["name", "description", "body"],
}


@R.register("lme_auto_memory_step")
class LmeAutoMemoryStep(BaseStep):
    """Extract each LME session into a daily note via a per-session agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reserve_lock = asyncio.Lock()
        self._reserved: dict[tuple[str, str], str] = {}

    def _resource_dir_name(self) -> str:
        return self.app_context.app_config.resource_dir if self.app_context is not None else "session"

    def _session_dir(self) -> Path:
        return self.workspace_path / self._resource_dir_name()

    def _concurrency(self) -> int:
        raw = self.context.get("concurrency") if self.context is not None else None
        if raw is None:
            raw = self.kwargs.get("concurrency", DEFAULT_CONCURRENCY)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_CONCURRENCY
        return value if value >= 1 else DEFAULT_CONCURRENCY

    @staticmethod
    def _parse_day(raw_date: str) -> str | None:
        """Parse a LongMemEval ``haystack_date`` (e.g. '2023/05/20 (Sat) 03:29') to YYYY-MM-DD."""
        head = raw_date.strip()[:10].replace("/", "-")
        return extract_daily_date(head)

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("session file is not a JSON object")
        return data

    @staticmethod
    def _format_messages(messages: list) -> str:
        lines: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).strip() or "unknown"
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            lines.append(f"[{role}]\n{content}")
        return "\n\n".join(lines)

    async def _existing_session_id(self, rel_path: str) -> str:
        note = self.workspace_path / rel_path
        if not note.is_file():
            return ""
        try:
            post = frontmatter.loads(note.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str((post.metadata or {}).get("session_id", "") or "").strip()

    async def _reserve_name(self, daily_dir: str, day: str, name: str, session_id: str) -> str:
        """Pick a collision-free filename stem for this session under ``day``."""
        async with self._reserve_lock:
            for cand in (name, f"{name}-{session_id}"):
                key = (day, cand)
                owner = self._reserved.get(key)
                if owner == session_id:
                    return cand
                if owner is not None:
                    continue
                existing = await self._existing_session_id(f"{daily_dir}/{day}/{cand}.md")
                if existing and existing != session_id:
                    continue
                self._reserved[key] = session_id
                return cand
            # Extremely unlikely fallback (same topic AND same session id twice).
            i = 2
            while True:
                cand = f"{name}-{session_id}-{i}"
                key = (day, cand)
                if key not in self._reserved:
                    self._reserved[key] = session_id
                    return cand
                i += 1

    async def execute(self):
        assert self.context is not None
        if self.agent_wrapper is None:
            raise ValueError("lme_auto_memory_step requires agent_wrapper")

        session_dir = self._session_dir()
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        session_files = sorted(p for p in session_dir.iterdir() if p.suffix == ".json")

        daily_dir = self.config_value("daily_dir")
        resource_dir = self._resource_dir_name()
        concurrency = self._concurrency()
        total = len(session_files)
        self.logger.info(
            f"[{self.name}] extracting {total} sessions from {session_dir} (concurrency={concurrency})",
        )

        self._reserved.clear()
        semaphore = asyncio.Semaphore(concurrency)

        async def extract_one(idx: int, session_path: Path) -> dict | None:
            try:
                session = self._load_json(session_path)
            except (ValueError, OSError):
                self.logger.exception(f"[{self.name}] skip {session_path.name}")
                return None

            session_id = str(session.get("haystack_session_id") or session_path.stem)
            if err := validate_session_id(session_id):
                self.logger.warning(f"[{self.name}] skip {session_path.name}: invalid session_id {err}")
                return None
            session_date = str(session.get("haystack_date") or "").strip()
            day = self._parse_day(session_date)
            if day is None:
                self.logger.warning(f"[{self.name}] skip {session_id}: unparseable date {session_date!r}")
                return None
            messages = session.get("messages") or []

            user_prompt = self.prompt_format(
                "user_message",
                session_id=session_id,
                session_date=session_date,
                messages=self._format_messages(messages),
            )
            async with semaphore:
                try:
                    result = await self.agent_wrapper.reply(
                        user_prompt,
                        system_prompt=self.get_prompt("system_prompt"),
                        output_schema=_MEMORY_SCHEMA,
                    )
                except Exception:  # noqa: BLE001 — one bad session must not abort the sweep
                    self.logger.exception(f"[{self.name}] extract failed for {session_id}")
                    return None

            extracted = result.get("structured_output")
            name = description = body = ""
            if isinstance(extracted, dict):
                name = str(extracted.get("name") or "").strip()
                description = str(extracted.get("description") or "").strip()
                body = str(extracted.get("body") or "").strip()
            if not isinstance(extracted, dict) or not name or not body:
                if isinstance(extracted, dict):
                    self.logger.info(f"[{self.name}] empty extraction for {session_id}; skipping")
                else:
                    self.logger.warning(f"[{self.name}] no structured output for {session_id}; skipping")
                return None

            unique_name = await self._reserve_name(daily_dir, day, name, session_id)
            rel_path = f"{daily_dir}/{day}/{unique_name}.md"
            post = frontmatter.Post(
                body,
                name=unique_name,
                description=description,
                session_id=session_id,
                session_date=session_date,
                source=f"[[{resource_dir}/{session_path.name}]]",
            )
            abs_path = self.workspace_path / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

            self.logger.info(f"[{self.name}] ({idx}/{total}) {session_id} -> {rel_path}")
            return {"session_id": session_id, "date": day, "path": rel_path}

        results = await asyncio.gather(
            *(extract_one(idx, path) for idx, path in enumerate(session_files, start=1)),
        )
        written = [r for r in results if r is not None]

        self.context.response.success = True
        self.context.response.answer = f"wrote {len(written)}/{total} session notes"
        self.context.response.metadata.update(
            {"num_sessions": total, "num_written": len(written), "notes": written},
        )
        return self.context.response
