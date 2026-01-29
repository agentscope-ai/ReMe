from loguru import logger

from ...core.op import BaseOp


class TestOp(BaseOp):

    async def execute(self):
        logger.info("delete start")
        # await self.vector_store.delete_all()
        await self.vector_store.delete("123")
        logger.info("delete end")
