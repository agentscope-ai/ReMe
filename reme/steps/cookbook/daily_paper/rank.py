"""Rank collected papers for the daily-paper workflow."""

from ....components import R
from ....schema import PaperInfo
from ._common import DailyPaperStep

_MEMORY_KEYWORDS = (
    "long-term memory",
    "long term memory",
    "lifelong memory",
    "agent memory",
    "episodic memory",
    "memory consolidation",
    "memory retrieval",
    "self-evolving memory",
    "continual learning",
    "context compression",
    "personalization",
    "knowledge graph",
    "retrieval augmented",
    "rag",
    "长期记忆",
    "记忆整合",
    "记忆检索",
)


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


def memory_keyword_score(paper: PaperInfo) -> int:
    """Return a lightweight recall score used to reserve memory-related candidates."""
    text = f"{paper.title}\n{paper.summary}".lower()
    return sum(keyword in text for keyword in _MEMORY_KEYWORDS)


def build_candidate_pool(papers: list[PaperInfo], *, limit: int = 20, memory_reserve: int = 5) -> list[PaperInfo]:
    """Keep strong general papers while reserving room for memory-related work."""
    if limit <= 0:
        raise ValueError("candidate_limit must be positive")
    ranked = sorted(papers, key=lambda item: (-item.fused_score, -item.upvotes, item.arxiv_id))
    reserve = min(max(memory_reserve, 0), limit)
    selected = ranked[: max(0, limit - reserve)]
    selected_ids = {paper.arxiv_id for paper in selected}
    memory_candidates = [
        paper for paper in ranked if paper.arxiv_id not in selected_ids and memory_keyword_score(paper)
    ]
    memory_candidates.sort(
        key=lambda item: (-memory_keyword_score(item), -item.fused_score, -item.upvotes, item.arxiv_id),
    )
    selected.extend(memory_candidates[:reserve])
    selected_ids = {paper.arxiv_id for paper in selected}
    selected.extend(paper for paper in ranked if paper.arxiv_id not in selected_ids and len(selected) < limit)
    return selected[:limit]


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
        memory_reserve = int(self._value("memory_reserve", 5))
        self.logger.info(
            f"[{self.name}] start papers={len(papers_by_id)} rrf_k={rrf_k} weekly_weight={weekly_weight} "
            f"candidate_limit={candidate_limit} memory_reserve={memory_reserve}",
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
            memory_reserve=memory_reserve,
        )
        if not candidates:
            raise RuntimeError("RRF produced no paper candidates")
        self._set_state("candidates", candidates)
        self.context.response.answer = f"Ranked {len(candidates)} paper candidates with RRF"
        self.logger.info(f"[{self.name}] finish candidates={len(candidates)}")
        return self.context.response
