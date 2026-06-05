"""Foreach dispatch: iterate changes and call a configured job per file."""

from ..base_step import BaseStep
from ...components import R


@R.register("foreach_dispatch_step")
class ForeachDispatchStep(BaseStep):
    """For each change item, call ``dispatch_job`` with the vault-relative path."""

    async def execute(self):
        assert self.context is not None
        changes: list[dict] = self.context.get("changes") or []
        dispatch_job: str = self.context.get("dispatch_job", "")
        if not dispatch_job:
            self.logger.warning(f"[{self.name}] no dispatch_job configured, skip")
            self.context.response.success = True
            return self.context.response
        for item in changes:
            rel_path = self.to_vault_relative(item["path"])
            try:
                await self.run_job(dispatch_job, file_path=rel_path, change=item["change"])
            except Exception:
                self.logger.exception(f"[{self.name}] dispatch {dispatch_job} failed: {rel_path}")
        self.context.response.success = True
        self.context.response.metadata["dispatched"] = len(changes)
        return self.context.response
