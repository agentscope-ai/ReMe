"""Select final daily papers with Claude Code."""

import json

from ....components import R
from ....schema import PaperInfo, PaperSelection
from ._common import DailyPaperStep, structured_output
from .rank import memory_keyword_score


@R.register("daily_paper_select_step")
class DailyPaperSelectStep(DailyPaperStep):
    """Use Claude Code to select the final papers."""

    @staticmethod
    def _validate_selection(selection: PaperSelection, candidates: list[PaperInfo], top_k: int) -> PaperSelection:
        candidate_ids = {paper.arxiv_id for paper in candidates}
        ordered = sorted(selection.selected, key=lambda item: item.rank)
        selected_ids = [item.arxiv_id for item in ordered]
        if len(ordered) != top_k:
            raise ValueError(f"Agent selected {len(ordered)} papers; expected {top_k}")
        if len(set(selected_ids)) != top_k or any(key not in candidate_ids for key in selected_ids):
            raise ValueError("Agent selection contains duplicate or out-of-pool ids")
        if [item.rank for item in ordered] != list(range(1, top_k + 1)):
            raise ValueError("Agent selection ranks must be consecutive starting at 1")
        alternates = [
            key for key in dict.fromkeys(selection.alternates) if key in candidate_ids and key not in selected_ids
        ]
        return selection.model_copy(update={"selected": ordered, "alternates": alternates})

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("Claude Code agent_wrapper is required for paper selection")
        candidates: list[PaperInfo] = self._state("candidates") or []
        top_k = int(self._value("top_k", 3))
        if top_k <= 0 or top_k > len(candidates):
            raise ValueError(f"top_k must be between 1 and {len(candidates)}")
        self.logger.info(f"[{self.name}] start candidates={len(candidates)} top_k={top_k}")

        candidate_payload = [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "summary": paper.summary,
                "authors": paper.authors,
                "organization": paper.organization,
                "upvotes": paper.upvotes,
                "monthly_rank": paper.monthly_rank,
                "weekly_rank": paper.weekly_rank,
                "fused_score": round(paper.fused_score, 8),
                "github_repo": paper.github_repo,
                "github_stars": paper.github_stars,
                "memory_keyword_score": memory_keyword_score(paper),
            }
            for paper in candidates
        ]
        feedback, selection = "", None
        for attempt in range(1, 3):
            self.logger.info(f"[{self.name}] agent start attempt={attempt}/2 candidates={len(candidates)}")
            result = await self.agent_wrapper.reply(
                self.prompt_format(
                    "select_user",
                    top_k=top_k,
                    candidates=json.dumps(candidate_payload, ensure_ascii=False, indent=2),
                    retry_feedback=feedback or "(none)",
                ),
                output_schema=PaperSelection,
            )
            try:
                selection = self._validate_selection(structured_output(result, PaperSelection), candidates, top_k)
                self.logger.info(f"[{self.name}] agent done attempt={attempt}/2 valid=True")
                break
            except (ValueError, TypeError) as exc:
                feedback = str(exc)
                self.logger.warning(
                    f"[{self.name}] agent done attempt={attempt}/2 valid=False error={feedback!r}",
                )
        if selection is None:
            raise RuntimeError(f"Claude Code paper selection failed validation: {feedback}")

        candidate_map = {paper.arxiv_id: paper for paper in candidates}
        selected_papers = [candidate_map[item.arxiv_id] for item in selection.selected]
        self._set_state("selection", selection)
        self._set_state("selected_papers", selected_papers)
        self.context.response.answer = f"Selected {top_k} papers with Claude Code"
        self.logger.info(
            f"[{self.name}] finish selected={','.join(item.arxiv_id for item in selection.selected)}",
        )
        return self.context.response
