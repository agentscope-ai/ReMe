"""Download and analyze selected daily-paper PDFs."""

import asyncio
import datetime as dt
import json
from pathlib import Path

from ....components import R
from ....schema import PaperInfo, PaperNoteOutput, PaperSelection, SelectedPaper
from ....utils.arxiv import ArxivPdfClient
from ._common import DailyPaperStep, strip_frontmatter, structured_output, write_markdown


@R.register("daily_paper_analyze_step")
class DailyPaperAnalyzeStep(DailyPaperStep):
    """Download each selected PDF and use Claude Code for detailed reading."""

    @staticmethod
    def _extract_pdf_text_sync(path: Path, max_pages: int, max_chars: int) -> tuple[str, int, bool]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency error has an explicit message
            raise RuntimeError("pypdf is required for the daily-paper workflow") from exc

        reader = PdfReader(str(path))
        chunks: list[str] = []
        size = 0
        page_count = min(len(reader.pages), max_pages)
        truncated = len(reader.pages) > max_pages
        for page_number, page in enumerate(reader.pages[:page_count], start=1):
            block = f"\n\n--- PAGE {page_number} ---\n\n{(page.extract_text() or '').strip()}"
            if size + len(block) > max_chars:
                if (remaining := max_chars - size) > 0:
                    chunks.append(block[:remaining])
                truncated = True
                break
            chunks.append(block)
            size += len(block)
        content = "".join(chunks).strip()
        if not content:
            raise ValueError(f"No extractable text found in PDF: {path.name}")
        return content, len(reader.pages), truncated

    async def _analyze_one(self, paper: PaperInfo, selected: SelectedPaper) -> tuple[str, str]:
        if self.agent_wrapper is None:
            raise RuntimeError("Claude Code agent_wrapper is required for paper analysis")
        day = self._run_day()
        daily_dir, resource_dir = (
            str(self.config_value("daily_dir")).strip("/"),
            str(self.config_value("resource_dir")).strip("/"),
        )
        pdf_rel, note_rel = (
            f"{resource_dir}/papers/{paper.arxiv_id}.pdf",
            f"{daily_dir}/{day}/paper-{paper.arxiv_id}.md",
        )
        pdf_path, note_path = self.workspace_path / pdf_rel, self.workspace_path / note_rel
        self.logger.info(f"[{self.name}] paper start arxiv_id={paper.arxiv_id}")

        downloader = ArxivPdfClient(
            timeout=float(self._value("pdf_timeout", 90.0)),
            max_bytes=int(self._value("max_pdf_bytes", 50 * 1024 * 1024)),
        )
        await downloader.download(paper.arxiv_id, pdf_path)
        self.logger.info(f"[{self.name}] pdf ready arxiv_id={paper.arxiv_id} path={pdf_rel}")
        pdf_text, page_count, truncated = await asyncio.to_thread(
            self._extract_pdf_text_sync,
            pdf_path,
            int(self._value("max_pdf_pages", 80)),
            int(self._value("max_pdf_chars", 240_000)),
        )
        self.logger.info(
            f"[{self.name}] pdf extracted arxiv_id={paper.arxiv_id} pages={page_count} "
            f"chars={len(pdf_text)} truncated={truncated}",
        )
        self.logger.info(f"[{self.name}] agent start arxiv_id={paper.arxiv_id}")
        result = await self.agent_wrapper.reply(
            self.prompt_format(
                "analyze_user",
                paper_info=json.dumps(paper.model_dump(), ensure_ascii=False, indent=2),
                selection_reason=selected.reason,
                memory_relevance=selected.memory_relevance,
                page_count=page_count,
                truncated=str(truncated).lower(),
                pdf_text=pdf_text,
            ),
            output_schema=PaperNoteOutput,
        )
        self.logger.info(f"[{self.name}] agent done arxiv_id={paper.arxiv_id}")
        output = structured_output(result, PaperNoteOutput)
        body = strip_frontmatter(output.body)
        if not output.description.strip() or not body:
            raise ValueError(f"Claude Code returned an empty paper note for {paper.arxiv_id}")
        await write_markdown(
            note_path,
            body,
            {
                "name": f"paper-{paper.arxiv_id}",
                "description": output.description.strip(),
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "authors": paper.authors,
                "hf_url": paper.hf_url,
                "arxiv_url": paper.arxiv_url,
                "download_url": paper.pdf_url,
                "source_pdf": f"[[{pdf_rel}]]",
                "published_at": paper.published_at,
                "monthly_rank": paper.monthly_rank,
                "weekly_rank": paper.weekly_rank,
                "fused_score": round(paper.fused_score, 8),
                "selection_reason": selected.reason,
                "memory_relevance": selected.memory_relevance,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "pdf_pages": page_count,
                "pdf_text_truncated": truncated,
            },
        )
        self.logger.info(f"[{self.name}] paper done arxiv_id={paper.arxiv_id} note_path={note_rel}")
        return note_rel, pdf_rel

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        selection: PaperSelection | None = self._state("selection")
        papers: list[PaperInfo] = self._state("selected_papers") or []
        if selection is None or len(selection.selected) != len(papers):
            raise RuntimeError("Paper selection state is missing before analysis")
        self.logger.info(f"[{self.name}] start papers={len(papers)}")

        note_paths, pdf_paths = [], []
        for paper, selected in zip(papers, selection.selected):
            note_path, pdf_path = await self._analyze_one(paper, selected)
            note_paths.append(note_path)
            pdf_paths.append(pdf_path)
        self._set_state("note_paths", note_paths)
        self._set_state("pdf_paths", pdf_paths)
        self.context.response.answer = f"Claude Code wrote {len(note_paths)} detailed paper notes"
        self.logger.info(f"[{self.name}] finish notes={len(note_paths)} pdfs={len(pdf_paths)}")
        return self.context.response
