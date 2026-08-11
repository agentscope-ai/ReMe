"""Rank collected papers for the daily-paper workflow."""

from ....components import R
from ....schema import PaperInfo
from ._common import DailyPaperStep


def rrf_score(
    monthly_rank: int | None,
    weekly_rank: int | None,
    *,
    rrf_k: int = 60,
    weekly_weight: float = 0.7,
) -> float:
    """Fuse optional monthly and weekly ranks with reciprocal-rank fusion."""
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    monthly_score = 0.0 if monthly_rank is None else 1.0 / (rrf_k + monthly_rank)
    weekly_score = 0.0 if weekly_rank is None else weekly_weight / (rrf_k + weekly_rank)
    return monthly_score + weekly_score


def build_candidate_pool(papers: list[PaperInfo], *, limit: int = 20) -> list[PaperInfo]:
    """Return the highest-ranked papers without applying a topic preference."""
    if limit <= 0:
        raise ValueError("candidate_limit must be positive")
    ranked = sorted(papers, key=lambda item: (-item.fused_score, -item.upvotes, item.arxiv_id))
    return ranked[:limit]


@R.register("daily_paper_rank_step")
class DailyPaperRankStep(DailyPaperStep):
    """Apply RRF and produce the bounded selection pool."""

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        papers_by_id: dict[str, PaperInfo] = self._state("info") or {}
        rrf_k, weekly_weight = int(self._value("rrf_k", 60)), float(self._value("weekly_weight", 0.7))
        candidate_limit = int(self._value("candidate_limit", 20))
        self.logger.info(
            f"[{self.name}] start papers={len(papers_by_id)} rrf_k={rrf_k} weekly_weight={weekly_weight} "
            f"candidate_limit={candidate_limit}",
        )
        for paper in papers_by_id.values():
            paper.fused_score = rrf_score(
                paper.monthly_rank,
                paper.weekly_rank,
                rrf_k=rrf_k,
                weekly_weight=weekly_weight,
            )
        candidates = build_candidate_pool(
            list(papers_by_id.values()),
            limit=candidate_limit,
        )
        if not candidates:
            raise RuntimeError("RRF produced no paper candidates")
        self._set_state("candidates", candidates)
        self.context.response.answer = f"Ranked {len(candidates)} paper candidates with RRF"
        self.logger.info(f"[{self.name}] finish candidates={len(candidates)}")
        return self.context.response
