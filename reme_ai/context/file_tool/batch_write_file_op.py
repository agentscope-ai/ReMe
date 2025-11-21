"""Batch write file operation module.

This module provides a tool operation for batch writing multiple files at once.
It processes a dictionary of file paths and contents, writing each file sequentially
and returning a combined result of all write operations.
"""

from flowllm.core.context import C
from flowllm.core.op import BaseAsyncOp
from flowllm.extensions.file_tool import WriteFileOp
from loguru import logger


@C.register_op()
class BatchWriteFileOp(BaseAsyncOp):
    """Batch write file operation.

    This operation writes multiple files in a single batch. It takes a dictionary
    of file paths and contents from the context, and writes each file using
    WriteFileOp. Returns a combined result of all write operations.
    """

    def __init__(self, save_answer: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_answer: bool = save_answer

    async def async_execute(self):
        """Execute the batch write file operation.

        Reads write_file_dict from context, which should be a dictionary mapping
        file paths to file contents. Writes each file sequentially and collects
        the results.
        """
        # Get write file dictionary from context
        write_file_dict: dict = self.context.get("write_file_dict", {})
        if not write_file_dict:
            self.context.response.answer = "No write file task."
            logger.info("No write file task.")
            return

        # Process each file in the dictionary
        result = []
        for file_path, content in write_file_dict.items():
            write_op = WriteFileOp(save_answer=self.save_answer)
            await write_op.async_call(file_path=file_path, content=content)
            result.append(write_op.output)

        # Combine all results into a single response
        self.context.response.answer = "\n".join(result)
