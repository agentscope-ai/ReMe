"""Portfolio-synthesis analysis step for Auto Fin."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .....components import R
from .....schema import PortfolioProposalOutput, PortfolioSnapshot, ProposedAction
from ._base import AutoFinAnalysisStep, json_text


class _PortfolioProposalDraft(BaseModel):
    """Agent-authored fields before deterministic ranking enrichment."""

    model_config = ConfigDict(extra="forbid")

    description: str
    body: str
    actions: list[ProposedAction] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def discard_generated_ranking(cls, value: Any) -> Any:
        """Ignore ranking copies because Quant owns the canonical ranking."""
        if isinstance(value, dict) and "fusion_ranking" in value:
            value = {key: item for key, item in value.items() if key != "fusion_ranking"}
        return value


@R.register("auto_fin_portfolio_step")
class AutoFinPortfolioStep(AutoFinAnalysisStep):
    """Combine upstream analyses into bounded portfolio proposals."""

    async def execute(self):
        run_context = self.state("run_context")
        snapshot = self.state("portfolio_snapshot")
        analyses = self.state("analyses")
        if (
            not isinstance(run_context, dict)
            or not isinstance(snapshot, PortfolioSnapshot)
            or not isinstance(analyses, dict)
        ):
            raise RuntimeError("Auto Fin portfolio inputs are missing before synthesis")
        self.require_checkpoint_reached(run_context)
        draft, error = await self.reply(
            "portfolio_user",
            _PortfolioProposalDraft,
            portfolio=json_text(snapshot.model_dump(mode="json")),
            analyses=json_text(analyses),
            run_context=json_text(run_context),
        )
        fusion = self.state("fusion_ranking")
        output = (
            PortfolioProposalOutput.model_validate(
                {**draft.model_dump(), "fusion_ranking": fusion},
            )
            if draft is not None
            else None
        )
        self.set_state("portfolio_output", output)
        self.set_state("portfolio_error", error)
        self.logger.info(
            f"[{self.name}] portfolio synthesis done valid={output is not None} "
            f"actions={len(output.actions) if output else 0}",
        )
        assert self.context is not None
        return self.context.response
