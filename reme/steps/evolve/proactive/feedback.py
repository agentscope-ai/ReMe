"""Proactive delivery feedback step: record what a host agent already surfaced.

The refresh chain decides what is worth raising; it cannot know what the host
actually raised. This step closes that loop by accepting a small report from
the consumer and appending it to the ``deliveries`` section of the truth source
``daily/_proactive.yaml``. Two readers benefit:

- ``proactive_step`` withholds agenda items still inside the cooldown window,
  so the same proposal is not surfaced twice;
- ``proactive_agenda_step`` suppresses those candidates deterministically and
  records an auditable reason instead of re-editing them into today's agenda.

Reporting is optional. A framework that never calls this step leaves
``deliveries`` empty and both readers behave exactly as they did before, which
keeps the contract additive rather than a behavioural change for every host.

The step is deliberately narrow: it never calls an LLM, never touches
``open_topics``/``resolved``/``schedules``, never rewrites ``interests.yaml``
and never advances a file catalog. It only appends feedback.
"""

import time

from ...base_step import BaseStep
from ....components import R
from ..dream.utils import daily_dir, today, workspace_dir
from .utils import (
    current_now,
    load_state,
    save_state,
    suppressed_delivery_ids,
    trim_deliveries,
    upsert_deliveries,
)


@R.register("proactive_feedback_step")
class ProactiveFeedbackStep(BaseStep):
    """Upsert host-reported agenda deliveries into the truth source.

    Records are idempotent on ``(topic_id, external_ref)`` when the host
    supplies a reference and on ``(topic_id, agenda_date, channel)``
    otherwise, so a retry of the same report updates in place. A terminal
    status (``executed``/``dismissed``) is never downgraded by a later
    ``shown`` report; ``failed`` never blocks re-delivery.
    """

    def __init__(
        self,
        retention_days: int = 180,
        max_deliveries: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.retention_days = max(int(retention_days), 1)
        self.max_deliveries = max(int(max_deliveries), 1)

    async def execute(self):
        assert self.context is not None
        started = time.monotonic()
        raw_items = self.context.get("items")
        items = list(raw_items) if isinstance(raw_items, (list, tuple)) else []
        day = today(self, str(self.context.get("date", "") or ""))
        daily = daily_dir(self)
        ws = workspace_dir(self)
        self.logger.info(f"[{self.name}] start day={day} items={len(items)}")

        if not items:
            self.context.response.success = True
            self.context.response.answer = "Feedback: no delivery items reported"
            self.context.response.metadata["proactive_feedback"] = {
                "date": day,
                "received": 0,
                "added": 0,
                "updated": 0,
                "dropped": 0,
                "trimmed": 0,
            }
            return self.context.response

        state_file, _needs_bootstrap = load_state(ws, daily)
        counts = upsert_deliveries(state_file, items, day, current_now(self))
        trimmed = trim_deliveries(state_file, day, self.retention_days, self.max_deliveries)
        await save_state(ws, state_file, daily)

        suppressed = suppressed_delivery_ids(state_file, day, self.retention_days)
        payload = {
            "date": day,
            "received": len(items),
            "added": counts["added"],
            "updated": counts["updated"],
            "dropped": counts["dropped"],
            "trimmed": trimmed,
            "total": len(state_file.deliveries),
            "suppressed_topics": sorted(suppressed),
        }
        self.context.response.success = True
        self.context.response.answer = (
            f"Feedback: added {counts['added']}, updated {counts['updated']}, "
            f"dropped {counts['dropped']}, trimmed {trimmed}; "
            f"{len(state_file.deliveries)} record(s) on file"
        )
        self.context.response.metadata["proactive_feedback"] = payload
        self.context.response.metadata["duration_ms"] = int((time.monotonic() - started) * 1000)
        self.logger.info(f"[{self.name}] finish {payload}")
        return self.context.response
