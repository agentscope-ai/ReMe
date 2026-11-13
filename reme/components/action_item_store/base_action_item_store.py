"""Abstract base class for action-item stores."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema import ActionItem, ActionItemQuery


class BaseActionItemStore(BaseComponent):
    """Persist action items and answer read-only queries about them.

    The store is the only owner of the records. Extraction lives in the memory
    workflow; reminders and delivery live in the host application. Queries must
    stay read-only: they must not trigger extraction, model calls or writes.
    """

    component_type = ComponentEnum.ACTION_ITEM_STORE

    async def load(self) -> None:
        """Load persisted records. No-op without local files."""

    async def dump(self) -> None:
        """Persist records. No-op without local files."""

    async def _start(self) -> None:
        await super()._start()
        await self.load()

    async def _close(self) -> None:
        if self.is_started:
            await self.dump()
        await super()._close()

    @abstractmethod
    async def upsert(self, items: list[ActionItem]) -> list[ActionItem]:
        """Insert new items or merge later observations into existing ones.

        Merging follows one rule: evidence observed earlier must not overwrite a
        resolution recorded later. Returns the stored items after the merge.
        """

    @abstractmethod
    async def get(self, item_id: str) -> ActionItem | None:
        """Return one item by id, or ``None`` when it is unknown."""

    @abstractmethod
    async def query(self, query: ActionItemQuery | None = None) -> list[ActionItem]:
        """Return items matching ``query``, newest update first."""
