"""Download and analyze selected daily-paper PDFs."""

import asyncio
import json
from pathlib import Path

from ....components import R
from ....schema import AnalyzedPaper, DailyPaperMarkdownOutput, PaperInfo, PaperPick
from ....utils.arxiv import ArxivPdfClient
from ._common import (
    PAPER_COUNT,
    DailyPaperStep,
    iter_note_metadata,
    normalize_chinese_title,
    replace_surrogates,
    resolve_unique_note_path,
    strip_frontmatter,
    structured_output,
    utc_now_iso,
    write_markdown,
)


@R.register("daily_paper_analyze_step")
class DailyPaperAnalyzeStep(DailyPaperStep):
    """Download and analyze the three papers selected for the daily brief."""

    @staticmethod
    def _extract_pdf_text_sync(
        path: Path,
        max_pages: int,
        max_chars: int,
    ) -> tuple[str, int, bool]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency error has an explicit message
            raise RuntimeError(
                "pypdf is required for the daily-paper workflow",
            ) from exc

        reader = PdfReader(str(path))
        chunks: list[str] = []
        size = 0
        page_count = min(len(reader.pages), max_pages)
        truncated = len(reader.pages) > max_pages
        for page_number, page in enumerate(reader.pages[:page_count], start=1):
            page_text = replace_surrogates((page.extract_text() or "").strip())
            block = f"\n\n--- PAGE {page_number} ---\n\n{page_text}"
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

    @staticmethod
    def _find_existing_note(day_dir: Path, arxiv_id: str) -> Path | None:
        """Find a prior generated note independently of its title filename."""
        for path, metadata in iter_note_metadata(day_dir):
            if metadata.get("arxiv_id") == arxiv_id and (
                metadata.get("kind") == "daily-paper-analysis" or path.name == f"paper-{arxiv_id}.md"
            ):
                return path
        return None

    async def _analyze_one(
        self,
        downloader: ArxivPdfClient,
        paper: PaperInfo,
        selected: PaperPick,
        used_titles: set[str],
    ) -> AnalyzedPaper:
        if self.agent_wrapper is None:
            raise RuntimeError("An agent_wrapper is required for paper analysis")
        day = self._run_day()
        daily_dir, resource_dir = (
            str(self.config_value("daily_dir")).strip("/"),
            str(self.config_value("resource_dir")).strip("/"),
        )
        pdf_rel = f"{resource_dir}/papers/{paper.arxiv_id}.pdf"
        pdf_path = self.workspace_path / pdf_rel
        self.logger.info(f"[{self.name}] paper start arxiv_id={paper.arxiv_id}")

        await downloader.download(paper.arxiv_id, pdf_path)
        self.logger.info(
            f"[{self.name}] pdf ready arxiv_id={paper.arxiv_id} path={pdf_rel}",
        )
        pdf_text, page_count, truncated = await asyncio.to_thread(
            self._extract_pdf_text_sync,
            pdf_path,
            int(self._value("max_pdf_pages", 35)),
            int(self._value("max_pdf_chars", 300_000)),
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
                selection_reason=selected.reasoning,
                page_count=page_count,
                truncated=str(truncated).lower(),
                pdf_text=pdf_text,
            ),
            output_schema=DailyPaperMarkdownOutput,
        )
        self.logger.info(f"[{self.name}] agent done arxiv_id={paper.arxiv_id}")
        output = structured_output(result, DailyPaperMarkdownOutput)
        title = normalize_chinese_title(output.title, f"论文解读-{paper.arxiv_id}")
        day_dir = self.workspace_path / daily_dir / day
        existing_note = self._find_existing_note(day_dir, paper.arxiv_id)
        suffix = f"（{paper.arxiv_id}）"
        title, note_path = resolve_unique_note_path(
            day_dir,
            title,
            taken=used_titles,
            taken_suffix=suffix,
            disk_suffix=suffix,
            existing=existing_note,
        )
        used_titles.add(title)
        note_rel = note_path.relative_to(self.workspace_path).as_posix()
        desc = replace_surrogates(output.desc.strip())
        body = replace_surrogates(strip_frontmatter(output.body))
        if not desc or not body:
            raise ValueError(f"Agent returned an empty paper note for {paper.arxiv_id}")
        await write_markdown(
            note_path,
            body,
            {
                "name": title,
                "title": title,
                "description": desc,
                "kind": "daily-paper-analysis",
                "arxiv_id": paper.arxiv_id,
                "source_title": paper.title,
                "authors": paper.authors,
                "hf_url": paper.hf_url,
                "arxiv_url": paper.arxiv_url,
                "download_url": paper.pdf_url,
                "source_pdf": f"[[{pdf_rel}]]",
                "published_at": paper.published_at,
                "monthly_rank": paper.monthly_rank,
                "weekly_rank": paper.weekly_rank,
                "fused_score": round(paper.fused_score, 8),
                "selection_reasoning": selected.reasoning,
                "generated_at": utc_now_iso(),
                "pdf_pages": page_count,
                "pdf_text_truncated": truncated,
            },
        )
        if existing_note is not None and existing_note != note_path:
            existing_note.unlink()
        self.logger.info(
            f"[{self.name}] paper done arxiv_id={paper.arxiv_id} note_path={note_rel}",
        )
        return AnalyzedPaper(
            arxiv_id=paper.arxiv_id,
            reasoning=selected.reasoning,
            title=title,
            desc=desc,
            body=body,
            note_path=note_rel,
            pdf_path=pdf_rel,
        )

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        selected: list[PaperPick] = self._state("selected") or []
        candidates: list[PaperInfo] = self._state("candidates") or []
        candidate_map = {paper.arxiv_id: paper for paper in candidates}
        if len(selected) != PAPER_COUNT or any(item.arxiv_id not in candidate_map for item in selected):
            raise RuntimeError("Paper selection state is missing before analysis")
        self.logger.info(f"[{self.name}] start papers={len(selected)}")

        analyses: list[AnalyzedPaper] = []
        used_titles: set[str] = set()
        async with ArxivPdfClient(
            timeout=float(self._value("pdf_timeout", 600.0)),
            max_bytes=int(self._value("max_pdf_bytes", 50 * 1024 * 1024)),
        ) as downloader:
            for item in selected:
                analyses.append(
                    await self._analyze_one(
                        downloader,
                        candidate_map[item.arxiv_id],
                        item,
                        used_titles,
                    ),
                )
        self._set_state("analyses", analyses)
        self.context.response.answer = f"Agent wrote {len(analyses)} detailed paper notes"
        self.logger.info(f"[{self.name}] finish notes={len(analyses)}")
        return self.context.response
