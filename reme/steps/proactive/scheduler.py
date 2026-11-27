"""Background Job that schedules the Manager's Proactive Runtimes."""

import asyncio

from .base import BaseAgentStep
from ...schema.proactive_agent import TIME_EVENT_SOURCE
from ...core.registry import R
from ...schema import Event


@R.register("agent_manager_loop_step")
class AgentManagerLoopStep(BaseAgentStep):
    """Own scheduler workers; Components retain resources and per-task state."""

    async def _work(self, runtime):
        try:
            response = await self.run_job("run_proactive_agent", agent_name=runtime.name)
            if not response.success:
                raise RuntimeError(str(response.answer))
            runtime.completed_round()
        except Exception as exc:
            runtime.failed_round(exc)
            self.logger.error(f"Task {runtime.name} failed: {exc}")
        finally:
            runtime.worker = None
            runtime.live_turns.notify()

    async def tick_runtime(self, runtime, definition):
        """Recover handoffs, admit due schedule input and start one eligible worker."""
        await runtime.recover_handoffs()
        if not definition.enabled or runtime.closed or runtime.retiring:
            runtime.next_trigger(definition)
            return
        now = runtime.now()
        if definition.agent_type == "time":
            scheduled = runtime.scheduled(definition, now)
            if scheduled is not None:
                try:
                    await runtime.inbox.append(
                        [
                            Event(
                                id=f"time-{runtime.name}-{int(scheduled.timestamp())}",
                                event_ts=scheduled,
                                source=TIME_EVENT_SOURCE,
                                description=f"Scheduled trigger: {scheduled.isoformat()}",
                            ),
                        ],
                        wait=False,
                    )
                except TimeoutError:
                    pass
                else:
                    runtime.advance_schedule(definition, now)
        reason, _ = runtime.waiting_state(definition, await runtime.inbox.snapshot())
        if reason or runtime.worker is not None or runtime.execution_lock.locked():
            return
        runtime.worker = asyncio.create_task(self._work(runtime), name=f"agent:{runtime.name}")

    async def tick(self):
        """Run one scheduling pass without storing state on the Step."""
        manager = self.agent_manager
        definitions = await manager.discover()
        for name, definition in definitions.items():
            try:
                runtime = await manager.resolve(name)
                await self.tick_runtime(runtime, definition)
            except (OSError, ValueError, RuntimeError) as exc:
                manager.registry.issues[name] = str(exc)
                self.logger.error(f"Task {name}: {exc}")

    async def execute(self):
        stop = self.context.stop_event
        if not isinstance(stop, asyncio.Event):
            raise RuntimeError("Agent scheduling requires a BackgroundJob stop_event")
        try:
            while not stop.is_set() and not self.agent_manager.closed:
                await self.tick()
                try:
                    await asyncio.wait_for(stop.wait(), timeout=1)
                except TimeoutError:
                    pass
        finally:
            workers = [runtime.worker for runtime in self.agent_manager.runtimes.values() if runtime.worker]
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        return self.context.response
