"""Common response handling for file operations."""

from ..base_step import BaseStep


class _FileIOStep(BaseStep):
    """Preserve the file-operation failure envelope without changing its payload."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        self.context.response.metadata.update(meta)
