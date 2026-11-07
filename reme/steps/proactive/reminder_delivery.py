"""Deliver pending reminders independently of proactive reasoning rounds."""

import asyncio

from ..base_step import BaseStep, Ref
from ...components import PaRuntime, R


@R.register("reminder_delivery_step")
class ReminderDeliveryStep(BaseStep):
    """Use the background Job lifecycle for delivery and ten-second idle checks."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("reminder_delivery requires an asyncio stop_event")
        while not stop.is_set():
            try:
                await self.pa.deliver_pending_reminder()
            except Exception:
                self.logger.exception("Pending reminder delivery failed")
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
        return self.context.response
