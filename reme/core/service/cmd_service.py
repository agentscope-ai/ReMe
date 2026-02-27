"""Command service module for managing and executing command-based workflows."""

from loguru import logger

from .base_service import BaseService
from ..flow import CmdFlow, BaseFlow
from ..utils import run_coro_safely


class CmdService(BaseService):
    """Service implementation for handling command flow execution logic."""

    def __init__(self, **kwargs):
        """Initialize the command service instance."""
        super().__init__(**kwargs)
        self._cmd_flow: CmdFlow | None = None

    def integrate_flow(self, flow: BaseFlow) -> str | None:
        """Integrate the workflow configuration into the command service."""
        self._cmd_flow = CmdFlow(flow=self.service_config.cmd.flow, service_context=self.service_context)
        return self._cmd_flow.tool_call.name if self._cmd_flow else None

    def run(self):
        """Execute the command flow in either asynchronous or synchronous mode."""
        super().run()
        if not self._cmd_flow:
            logger.warning("No command flow configured, skipping execution")
            return
        kwargs = self.service_config.cmd.model_extra
        if self._cmd_flow.async_mode:

            async def async_run():
                await self.service_context.start()
                return await self._cmd_flow.call(**kwargs)

            response = run_coro_safely(async_run())
        else:
            run_coro_safely(self.service_context.start())
            response = self._cmd_flow.call_sync(**kwargs)

        if response.answer:
            logger.info(f"response.answer={response.answer}")
