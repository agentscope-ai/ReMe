"""Log changes step: mock dispatch target that logs detected changes."""

from ..base_step import BaseStep
from ...components import R


@R.register("log_changes_step")
class LogChangesStep(BaseStep):
    """Log each change item. Placeholder for future digest-watch logic."""

    async def execute(self):
        assert self.context is not None
        changes: list[dict] = self.context.get("changes") or []
        for item in changes:
            self.logger.info(f"[{self.name}] {item['change']}: {item['path']}")
        self.context.response.success = True
        self.context.response.metadata["count"] = len(changes)
        return self.context.response
