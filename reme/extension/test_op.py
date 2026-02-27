"""Test workflow operations."""

from loguru import logger

from ..core.op import BaseOp


class TestOp(BaseOp):
    """Test operation for workflow testing."""

    async def execute(self):
        logger.info("delete start")
        # await self.vector_store.delete_all()
        await self.vector_store.delete("123")
        logger.info("delete end")
