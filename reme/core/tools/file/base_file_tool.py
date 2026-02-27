"""Base class for file system tools with unified error handling."""

from loguru import logger

from ...op import BaseTool
from ...runtime_context import RuntimeContext


class BaseFileTool(BaseTool):
    """Base class for file system tools.

    Features:
    - No retry logic (max_retries=1)
    - Catches all exceptions and returns error messages to LLM
    - Simplifies error handling in subclasses
    """

    def __init__(self, **kwargs):
        """Initialize fs tool with no retry."""
        kwargs.setdefault("max_retries", 1)
        kwargs.setdefault("raise_exception", False)
        super().__init__(**kwargs)

    async def call(self, context: RuntimeContext = None, **kwargs):
        """Execute the tool with unified error handling.

        This method catches all exceptions and returns error messages
        to the LLM instead of raising them.
        """
        self.context = RuntimeContext.from_context(context, **kwargs)

        try:
            await self.before_execute()
            response = await self.execute()
            response = await self.after_execute(response)
            return response

        except Exception as e:
            # Return error message to LLM instead of raising
            error_msg = f"{self.__class__.__name__} failed: {str(e)}"
            logger.error(error_msg)
            return await self.after_execute(error_msg)
