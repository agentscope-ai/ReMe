"""Command-line service for executing a single configured flow.

This service runs a specified flow once, synchronously or asynchronously,
logging the final answer if present.
"""

import asyncio

from loguru import logger

from .base_service import BaseService
from ..context import C
from ..flow import CmdFlow


@C.register_service("cmd")
class CmdService(BaseService):
    """Service that executes a configured command flow and exits."""

    def run(self):
        """Execute the configured flow and print the result if available."""
        super().run()
        cmd_config = self.service_config.cmd
        flow = CmdFlow(flow=cmd_config.flow)
        if flow.async_mode:
            response = asyncio.run(flow.async_call(**self.service_config.cmd.params))
        else:
            response = flow.call(**self.service_config.cmd.params)

        if response.answer:
            logger.info(f"response.answer={response.answer}")
