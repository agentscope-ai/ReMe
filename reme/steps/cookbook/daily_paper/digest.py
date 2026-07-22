"""Build the final daily-paper brief from detailed notes."""

import datetime as dt
import json
from types import SimpleNamespace

from ....components import R
from ....schema import DailyBriefOutput, PaperSelection
from ...file_io import refresh_day_index
from ._common import DailyPaperStep, strip_frontmatter, structured_output, write_markdown


@R.register("daily_paper_digest_step")
class DailyPaperDigestStep(DailyPaperStep):
    """Use Claude Code to read the detailed notes and create the final brief."""

    async def execute(self):
        assert self.context is not None
        if self._skip():
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("Claude Code agent_wrapper is required for the daily brief")
        note_paths: list[str] = self._state("note_paths") or []
        selection: PaperSelection | None = self._state("selection")
        if selection is None or not note_paths:
            raise RuntimeError("Detailed paper notes are missing before digest generation")

        absolute_paths = [str((self.workspace_path / path).resolve()) for path in note_paths]
        wikilinks = [f"[[{path}]]" for path in note_paths]
        result = await self.agent_wrapper.reply(
            self.prompt_format(
                "digest_user",
                top_k=len(note_paths),
                note_paths=json.dumps(absolute_paths, ensure_ascii=False, indent=2),
                wikilinks=json.dumps(wikilinks, ensure_ascii=False, indent=2),
            ),
            output_schema=DailyBriefOutput,
        )
        output = structured_output(result, DailyBriefOutput)
        body = strip_frontmatter(output.body)
        if not output.description.strip() or not body:
            raise ValueError("Claude Code returned an empty daily paper brief")
        missing_links = [link for link in wikilinks if link not in body]
        if missing_links:
            body += "\n\n## 详细文章\n\n" + "\n".join(f"- {link}" for link in missing_links)

        day = self._run_day()
        daily_dir = str(self.config_value("daily_dir")).strip("/")
        digest_rel = f"{daily_dir}/{day}/daily-paper-brief.md"
        selected_ids = [item.arxiv_id for item in selection.selected]
        await write_markdown(
            self.workspace_path / digest_rel,
            body,
            {
                "name": "daily-paper-brief",
                "description": output.description.strip(),
                "date": day,
                "arxiv_ids": selected_ids,
                "source_notes": wikilinks,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        await refresh_day_index(SimpleNamespace(workspace_path=self.workspace_path), day, daily_dir)

        manifest = dict(self._state("manifest") or {})
        manifest.update(
            {
                "status": "complete",
                "note_paths": note_paths,
                "pdf_paths": self._state("pdf_paths"),
                "digest_path": digest_rel,
            },
        )
        manifest_path = await self._write_manifest(manifest)
        self.context.response.success = True
        self.context.response.answer = f"Generated daily paper brief: {digest_rel}"
        self.context.response.metadata.update(
            {
                "date": day,
                "week": self._state("week"),
                "month": self._state("month"),
                "selection_reasoning": selection.selection_reasoning,
                "selected_arxiv_ids": selected_ids,
                "note_paths": note_paths,
                "pdf_paths": self._state("pdf_paths"),
                "digest_path": digest_rel,
                "manifest_path": str(manifest_path.relative_to(self.workspace_path)),
                "source_counts": self._state("source_counts"),
                "excluded_yesterday_count": len(self._state("excluded_yesterday") or []),
                "excluded_history_count": len(self._state("excluded_history") or []),
            },
        )
        return self.context.response
