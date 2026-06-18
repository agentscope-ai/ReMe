"""Read daily topic notes for proactive use."""

import re
from pathlib import Path

import frontmatter
from pydantic import BaseModel, Field

from ._evolve import now
from ..base_step import BaseStep
from ..file_io._daily_index import validate_session_id
from ...components import R


class ProactiveResult(BaseModel):
    """Result of reading the latest daily topic note."""

    date: str = ""
    path: str = ""
    topics: list[dict] = Field(default_factory=list)
    content: str = ""
    skipped: bool = False
    error: str = ""
    summary: str = ""


def _parse_topics(content: str) -> list[dict]:
    topics: list[dict] = []
    current: dict | None = None
    for line in content.splitlines():
        match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*\s*$", line)
        if match:
            if current:
                topics.append(current)
            current = {"title": match.group(1).strip(), "reason": "", "evidence": "", "keywords": []}
            continue
        if current is None:
            continue
        stripped = line.strip()
        if stripped.startswith("- reason:"):
            current["reason"] = stripped.removeprefix("- reason:").strip()
        elif stripped.startswith("- evidence:"):
            current["evidence"] = stripped.removeprefix("- evidence:").strip()
        elif stripped.startswith("- keywords:"):
            raw = stripped.removeprefix("- keywords:").strip()
            current["keywords"] = [part.strip() for part in raw.split(",") if part.strip()]
    if current:
        topics.append(current)
    return topics


@R.register("proactive_step")
class ProactiveStep(BaseStep):
    """Read the daily topics markdown produced by ``daily_topics_step``."""

    def __init__(
        self,
        session_id: str = "interests",
        include_content: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session_id = session_id
        self.include_content = include_content

    def _daily_dir(self) -> str:
        cfg = self.app_context.app_config if self.app_context is not None else None
        return (cfg.daily_dir if cfg else "") or "daily"

    def _vault_dir(self) -> Path:
        vr = getattr(self.file_store, "vault_path", None)
        return Path(vr).resolve() if vr else Path.cwd().resolve()

    async def execute(self):
        assert self.context is not None
        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        day = (self.context.get("date", "") or "").strip() or now(tz).strftime("%Y-%m-%d")
        session_id = (self.context.get("session_id", self.session_id) or self.session_id).strip()
        include_content = bool(self.context.get("include_content", self.include_content))

        result = ProactiveResult(date=day)
        err = validate_session_id(session_id)
        if err:
            result.error = err
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            self.context.response.metadata.update(result.model_dump())
            return self.context.response

        rel_path = f"{self._daily_dir()}/{day}/session_agent_{session_id}.md"
        result.path = rel_path
        abs_path = self._vault_dir() / rel_path
        if not abs_path.is_file():
            result.skipped = True
            result.summary = f"Skipped: daily topics note not found at {rel_path}"
            self.context.response.success = True
            self.context.response.answer = result.summary
            self.context.response.metadata.update(result.model_dump())
            return self.context.response

        try:
            text = abs_path.read_text(encoding="utf-8")
            post = frontmatter.loads(text)
        except Exception as e:  # noqa: BLE001
            result.error = f"{type(e).__name__}: {e}"
            self.context.response.success = False
            self.context.response.answer = f"Error: {result.error}"
            self.context.response.metadata.update(result.model_dump())
            return self.context.response

        result.topics = _parse_topics(post.content)
        if include_content:
            result.content = text
        result.summary = f"Read {len(result.topics)} proactive topic(s) from {rel_path}"
        self.context.response.success = True
        self.context.response.answer = result.summary
        self.context.response.metadata.update(result.model_dump())
        return self.context.response
