"""Browse Proactive inputs without consuming or modifying them."""

from ..base import BasePaStep
from ....components import R


@R.register("proactive_inbox_view_step")
class ProactiveInboxViewStep(BasePaStep):
    """Expose the shared projection without inheriting another transport Step."""

    async def execute(self):
        self.require_paths(
            self.pa.tasks.root_path,
            self.pa.cron_path,
            self.pa.proactive_inbox_path,
            self.pa.inbox.cursor_path,
            self.pa.inbox.receipts_path,
        )
        self.context.response.answer = await self.pa.views.inbox(self.context.data)
        return self.context.response
