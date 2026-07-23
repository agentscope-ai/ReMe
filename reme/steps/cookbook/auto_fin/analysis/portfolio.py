"""Portfolio-synthesis analysis step for Auto Fin."""

from __future__ import annotations

from .....components import R
from .....schema import PortfolioProposalOutput, PortfolioSnapshot
from ._base import AutoFinAnalysisStep, json_text


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
        output, error = await self.reply(
            "portfolio_user",
            PortfolioProposalOutput,
            portfolio=json_text(snapshot.model_dump(mode="json")),
            analyses=json_text(analyses),
            run_context=json_text(run_context),
        )
        fusion = self.state("fusion_ranking")
        if output is not None and fusion is not None:
            output = output.model_copy(update={"fusion_ranking": fusion})
        self.set_state("portfolio_output", output)
        self.set_state("portfolio_error", error)
        self.logger.info(
            f"[{self.name}] portfolio synthesis done valid={output is not None} "
            f"actions={len(output.actions) if output else 0}",
        )
        assert self.context is not None
        return self.context.response
