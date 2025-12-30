import asyncio

from loguru import logger

from .base_service import BaseService
from ..context import C
from ..flow import CmdFlow


@C.register_service("cmd")
class CmdService(BaseService):

    def run(self):
        super().run()
        cmd_config = self.service_config.cmd
        flow = CmdFlow(flow=cmd_config.flow)
        if flow.async_mode:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(flow.call(**self.service_config.cmd.model_extra))
            else:
                import nest_asyncio

                nest_asyncio.apply()
                asyncio.run(flow.call(**self.service_config.cmd.model_extra))

        else:
            response = flow.call_sync(**self.service_config.cmd.model_extra)

        if response.answer:
            logger.info(f"response.answer={response.answer}")
