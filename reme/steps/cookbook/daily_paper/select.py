"""Select final daily papers with an agent."""

import json

from ....components import R
from ....schema import PaperInfo, PaperPick, PaperPickList
from ._common import PAPER_COUNT, DailyPaperStep, structured_output

_MAX_SELECT_ATTEMPTS = 2


@R.register("daily_paper_select_step")
class DailyPaperSelectStep(DailyPaperStep):
    """Use an agent to select the final papers."""

    @staticmethod
    def _validate_selection(
        output: PaperPickList,
        candidates: list[PaperInfo],
    ) -> list[PaperPick]:
        """Validate exactly three unique, in-pool selections."""
        candidate_ids = {paper.arxiv_id for paper in candidates}
        if len(output.papers) != PAPER_COUNT:
            raise ValueError(
                f"Agent selected {len(output.papers)} papers; expected {PAPER_COUNT}",
            )
        selected = [
            PaperPick(arxiv_id=item.arxiv_id.strip(), reasoning=item.reasoning.strip()) for item in output.papers
        ]
        selected_ids = [item.arxiv_id for item in selected]
        if len(set(selected_ids)) != PAPER_COUNT:
            raise ValueError("Agent selection contains duplicate arxiv_ids")
        if any(arxiv_id not in candidate_ids for arxiv_id in selected_ids):
            raise ValueError("Agent returned an arxiv_id outside the candidate pool")
        if any(not item.reasoning for item in selected):
            raise ValueError("Agent selection reasoning cannot be empty")
        return selected

    async def execute(self):
        assert self.context is not None
        if self._skip():
            self.logger.info(f"[{self.name}] skip existing digest")
            return self.context.response
        if self.agent_wrapper is None:
            raise RuntimeError("An agent_wrapper is required for paper selection")
        candidates: list[PaperInfo] = self._state("candidates") or []
        topics = str(self._value("topics", "") or "").strip()
        selection_preference = (
            f"用户明确感兴趣的主题：{topics}\n仅将这些 topics 作为主题偏好。"
            if topics
            else (
                "用户未提供明确的 topic 倾向。优先选择 fused_score 更高的论文；"
                "只有在研究价值、新颖性、影响或可读性明显更强时才偏离分数排序，"
                "并在 reasoning 中具体说明相对高分候选的优势。"
            )
        )
        if len(candidates) < PAPER_COUNT:
            raise ValueError(
                f"At least {PAPER_COUNT} paper candidates are required for selection",
            )
        self.logger.info(f"[{self.name}] start candidates={len(candidates)}")

        candidate_payload = [
            {
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "summary": paper.summary,
                "upvotes": paper.upvotes,
                "monthly_rank": paper.monthly_rank,
                "weekly_rank": paper.weekly_rank,
                "fused_score": round(paper.fused_score, 8),
            }
            for paper in candidates
        ]
        feedback, selected = "", None
        for attempt in range(1, _MAX_SELECT_ATTEMPTS + 1):
            self.logger.info(
                f"[{self.name}] agent start attempt={attempt}/{_MAX_SELECT_ATTEMPTS} candidates={len(candidates)}",
            )
            result = await self.agent_wrapper.reply(
                self.prompt_format(
                    "select_user",
                    candidates=json.dumps(
                        candidate_payload,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    retry_feedback=feedback or "(none)",
                    selection_preference=selection_preference,
                ),
                output_schema=PaperPickList,
            )
            try:
                selected = self._validate_selection(
                    structured_output(result, PaperPickList),
                    candidates,
                )
                self.logger.info(
                    f"[{self.name}] agent done attempt={attempt}/{_MAX_SELECT_ATTEMPTS} valid=True",
                )
                break
            except (ValueError, TypeError) as exc:
                feedback = str(exc)
                self.logger.warning(
                    f"[{self.name}] agent done attempt={attempt}/{_MAX_SELECT_ATTEMPTS} valid=False error={feedback!r}",
                )
        if selected is None:
            raise RuntimeError(f"Agent paper selection failed validation: {feedback}")

        self._set_state("selected", selected)
        selected_ids = [item.arxiv_id for item in selected]
        self.context.response.answer = f"Selected {PAPER_COUNT} papers with an agent"
        self.logger.info(f"[{self.name}] finish selected={','.join(selected_ids)}")
        return self.context.response
