"""Serial proactive wakeups controlled by the user-owned JSON configuration."""

# pylint: disable=protected-access

import asyncio

from ..base_step import BaseStep, Ref
from ...components import PaRuntime, R


@R.register("pa_loop_step")
class PaLoopStep(BaseStep):
    """Recheck active hours and edited intervals while waiting for the next round."""

    pa: PaRuntime = Ref(PaRuntime, "pa")

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("pa_loop_step requires an asyncio stop_event")
        anchor = self.pa._now()
        while not stop.is_set():
            try:
                config = await self.pa.get_proactive_config()
            except (OSError, ValueError) as exc:
                self.logger.error(f"Cannot load proactive configuration: {exc}")
                delay = 30
            else:
                now = self.pa._now()
                wakeup = config.next_wakeup(anchor) if config.is_active(now) else config.next_start(now)
                delay = (wakeup - now).total_seconds()
                if delay <= 0:
                    await self.run_job(self.kwargs.get("job", "pa_proactive"), stop_event=stop)
                    anchor = self.pa._now()
                    continue
                delay = min(delay, 30)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
        return self.context.response
