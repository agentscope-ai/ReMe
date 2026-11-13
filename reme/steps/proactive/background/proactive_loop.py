"""Serial proactive wakeups controlled by the user-owned JSON configuration."""

# pylint: disable=protected-access

from ._lifecycle import require_stop_event, wait_for_stop

from ..base import BasePaStep
from ....components import R


@R.register("pa_loop_step")
class PaLoopStep(BasePaStep):
    """Recheck active hours and edited intervals while waiting for the next round."""

    async def execute(self):
        stop = require_stop_event(self.context.stop_event)
        anchor = self.pa.now()
        while not stop.is_set():
            delay = 30
            try:
                config = await self.pa.get_proactive_config()
                reason = await self.pa.proactive_skip_reason(config)
            except (OSError, ValueError) as exc:
                self.logger.error(f"Cannot load proactive configuration: {exc}")
            else:
                if not reason:
                    delay = (config.next_wakeup(anchor) - self.pa.now()).total_seconds()
                    if delay <= 0:
                        await self.run_job(self.kwargs.get("job", "pa_proactive"), stop_event=stop)
                        anchor = self.pa.now()
                        continue
                    delay = min(delay, 30)
            await wait_for_stop(stop, delay)
        return self.context.response
