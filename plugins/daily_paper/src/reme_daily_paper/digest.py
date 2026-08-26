"""Build the final daily-paper brief from detailed notes."""

import datetime as dt
import json
import re
from pathlib import Path
from types import SimpleNamespace

from reme.steps.file_io import refresh_day_index

from .base import (
    PAPER_COUNT,
    DailyPaperStep,
    normalize_chinese_title,
    resolve_unique_note_path,
    strip_frontmatter,
    structured_output,
    utc_now_iso,
    write_markdown,
)
from .schema import AnalyzedPaper, DailyPaperMarkdownOutput

_WIKILINK_RE = re.compile(r"\[\[([^\[\]\n]+)\]\]")


class DailyPaperDigestStep(DailyPaperStep):
    """Use an agent to read the detailed notes and create the final brief."""

    def _valid_historical_wikilink(self, path: str, run_day: dt.date, digest: Path) -> bool:
        """Return whether one path is a safe, existing note from an earlier day."""
        parts = path.split("/")
        if not path or path.startswith("/") or "\\" in path or not path.endswith(".md"):
            return False
        if any(part in {"", ".", ".."} for part in parts) or any(character in path for character in "[]|"):
            return False
        daily_dir = str(self.config_value("daily_dir")).strip("/")
        if not path.startswith(f"{daily_dir}/"):
            return False
        relative_parts = path[len(daily_dir) + 1 :].split("/")
        if len(relative_parts) < 2:
            return False
        try:
            linked_day = dt.date.fromisoformat(relative_parts[0])
            resolved = (self.workspace_path.resolve() / path).resolve()
            resolved.relative_to(self.workspace_path.resolve())
        except (ValueError, OSError):
            return False
        return linked_day < run_day and resolved.is_file() and resolved != digest

    def _validate_historical_wikilinks(self, body: str, run_day: dt.date, digest_path: Path) -> str:
        """Keep only real historical daily-note links emitted by the agent."""
        digest = digest_path.resolve()

        def replace(match: re.Match[str]) -> str:
            inner = match.group(1).strip()
            raw_target, separator, raw_alias = inner.partition("|")
            target = raw_target.strip()
            path = target.partition("#")[0].strip()
            alias = (raw_alias.strip() if separator else "") or Path(path).stem.replace("_", " ")
            if not self._valid_historical_wikilink(path, run_day, digest):
                return alias
            return match.group(0)

        return _WIKILINK_RE.sub(replace, body)

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("An agent_wrapper is required for the daily brief")
        analyses: list[AnalyzedPaper] = self._state("analyses") or []
        if len(analyses) != PAPER_COUNT:
            raise RuntimeError(
                "Detailed paper notes are missing before digest generation",
            )
        self.logger.info(f"[{self.name}] start notes={len(analyses)}")

        documents = [{"title": item.title, "desc": item.desc, "body": item.body} for item in analyses]
        wikilinks = [f"[[{item.note_path}]]" for item in analyses]
        previous_day = (dt.date.fromisoformat(self._run_day()) - dt.timedelta(days=1)).isoformat()
        self.logger.info(f"[{self.name}] agent start notes={len(analyses)}")
        result = await self.agent_wrapper.reply(
            self.prompt_format(
                "digest_user",
                documents=json.dumps(documents, ensure_ascii=False, indent=2),
                previous_day=previous_day,
            ),
            output_schema=DailyPaperMarkdownOutput,
            job_tools=list(self.kwargs.get("job_tools") or []),
        )
        self.logger.info(f"[{self.name}] agent done notes={len(analyses)}")
        output = structured_output(result, DailyPaperMarkdownOutput)
        body = strip_frontmatter(output.body)
        if not output.desc.strip() or not body:
            raise ValueError("Agent returned an empty daily paper brief")

        day = self._run_day()
        daily_dir = str(self.config_value("daily_dir")).strip("/")
        title = normalize_chinese_title(output.title, f"每日论文简报-{day}")
        existing_rel = str(self._state("existing_digest_path") or "").strip()
        existing_path = self.workspace_path / existing_rel if existing_rel else None
        title, digest_path = resolve_unique_note_path(
            self.workspace_path / daily_dir / day,
            title,
            taken={item.title for item in analyses},
            taken_suffix="（每日简报）",
            disk_suffix=f"（{day}）",
            existing=existing_path,
        )
        digest_rel = digest_path.relative_to(self.workspace_path).as_posix()
        body = self._validate_historical_wikilinks(body, dt.date.fromisoformat(day), digest_path)
        body += "\n\n## 详细论文\n\n" + "\n".join(f"- {link}" for link in wikilinks)
        selected_ids = [item.arxiv_id for item in analyses]
        await write_markdown(
            digest_path,
            body,
            {
                "name": title,
                "title": title,
                "description": output.desc.strip(),
                "kind": "daily-paper-brief",
                "date": day,
                "arxiv_ids": selected_ids,
                "selection_reasoning": [item.reasoning for item in analyses],
                "source_notes": wikilinks,
                "generated_at": utc_now_iso(),
            },
        )
        if existing_path is not None and existing_path != digest_path:
            existing_path.unlink()
        self._set_state("digest_path", digest_rel)
        self.logger.info(f"[{self.name}] digest written path={digest_rel}")
        self.logger.info(
            f"[{self.name}] refresh index start date={day} daily_dir={daily_dir}",
        )
        await refresh_day_index(
            SimpleNamespace(workspace_path=self.workspace_path),
            day,
            daily_dir,
        )
        self.logger.info(f"[{self.name}] refresh index done date={day}")

        self.context.response.success = True
        self.context.response.answer = f"Generated daily paper brief: {digest_rel}"
        self.context.response.metadata.update(
            {
                "date": day,
                "week": self._state("week"),
                "month": self._state("month"),
                "selection_reasoning": [item.reasoning for item in analyses],
                "selected_arxiv_ids": selected_ids,
                "note_paths": [item.note_path for item in analyses],
                "pdf_paths": [item.pdf_path for item in analyses],
                "digest_path": digest_rel,
                "source_counts": self._state("source_counts"),
                "excluded_yesterday_count": len(
                    self._state("excluded_yesterday") or [],
                ),
                "excluded_history_count": len(self._state("excluded_history") or []),
            },
        )
        self.logger.info(
            f"[{self.name}] finish date={day} papers={len(selected_ids)} digest_path={digest_rel}",
        )
        return self.context.response
