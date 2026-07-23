"""Idempotent DingTalk notification delivery for Auto Fin reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from ....components import R
from ..dingtalk.send import DingTalkMarkdownSendStep
from ._common import write_atomic


@R.register("auto_fin_notification_step")
class AutoFinNotificationStep(DingTalkMarkdownSendStep):
    """Send at most one successful notification per Auto Fin run stage."""

    async def execute(self):
        assert self.context is not None
        if not self.context.response.metadata.get("notify", False):
            self.logger.info(
                f"[{self.name}] skip Auto Fin notification for a non-notifying run",
            )
            return self.context.response
        run_id = str(self.context.response.metadata.get("run_id") or "")
        if not run_id:
            raise RuntimeError("Auto Fin notification requires run_id")
        stage = str(self.context.get("auto_fin_notification_stage") or "").strip()
        if stage and stage not in {"event", "backtest", "us_correlation", "portfolio"}:
            raise ValueError(f"unsupported Auto Fin notification stage: {stage}")
        state_name = f"{run_id}.{stage}.json" if stage else f"{run_id}.json"

        state_path = (
            self.workspace_path
            / str(self.config_value("metadata_dir"))
            / "auto-fin"
            / "notification-state"
            / state_name
        )
        force_notify = bool(
            self.context.get("force_notify", self.kwargs.get("force_notify", False)),
        )
        if state_path.is_file() and not force_notify:
            self.context.response.metadata["dingtalk_skipped_duplicate"] = True
            self.context.response.metadata["dingtalk_sent_count"] = 0
            self.logger.info(
                f"[{self.name}] skip duplicate Auto Fin notification run_id={run_id}",
            )
            return self.context.response

        self.logger.info(
            f"[{self.name}] Auto Fin notification start run_id={run_id} "
            f"stage={stage or 'final'} force={force_notify}",
        )
        response = await super().execute()
        sent_count = int(response.metadata.get("dingtalk_sent_count", 0))
        if sent_count:
            await write_atomic(
                state_path,
                json.dumps(
                    {
                        "run_id": run_id,
                        "stage": stage or "final",
                        "sent_at": datetime.now(timezone.utc).isoformat(),
                        "sent_count": sent_count,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            self.logger.info(
                f"[{self.name}] Auto Fin notification recorded run_id={run_id} "
                f"stage={stage or 'final'} sent_count={sent_count}",
            )
        else:
            self.logger.info(
                f"[{self.name}] Auto Fin notification completed without delivery run_id={run_id}",
            )
        return response
