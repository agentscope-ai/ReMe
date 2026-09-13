"""Paginated listing of active file tags."""

from ..base_step import BaseStep
from ...components import R


@R.register("list_tags_step")
class ListTagsStep(BaseStep):
    """Return one compact page of active tags and their indexed file counts."""

    async def execute(self):
        assert self.context is not None
        tag_index = self.file_store.require_tag_index()

        result = await tag_index.list_tags(
            page=self.context.get("page", 1),
            order_by=self.context.get("order_by", "tag"),
            order=self.context.get("order"),
            page_size=self.context.get("page_size", 100),
        )
        self.context.response.answer = result
        return self.context.response
