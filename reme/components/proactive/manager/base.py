"""Background Runtime ownership and event routing contracts."""

from abc import abstractmethod
import asyncio

from ...base import BaseComponent
from ..inbox import InvalidInboxBatchError
from ....enumeration import ComponentEnum


class TaskNotFoundError(ValueError):
    """An identity is missing, deleted, or being retired."""


class TimeTriggeredTaskDeliveryError(ValueError):
    """External observations cannot enter a time-triggered task."""


class BaseAgentManager(BaseComponent):
    """Manage background identities; Remy remains a separate Runtime."""

    component_type = ComponentEnum.AGENT_MANAGER

    @abstractmethod
    async def resolve(self, agent_name):
        """Return the one static reference or owned dynamic Runtime."""

    @abstractmethod
    async def definition(self, agent_name, *, include_deleted=False):
        """Read the user-owned definition."""

    @abstractmethod
    async def create(self, config):
        """Create or restore a persistent identity."""

    @abstractmethod
    async def update(self, agent_name, settings):
        """Publish validated runtime settings."""

    @abstractmethod
    async def delete(self, agent_name):
        """Retire execution while preserving its workspace files."""

    @abstractmethod
    async def append_events(self, agent_name, events):
        """Append to one receiver independently."""

    async def deliver_events(self, task_names, events):
        """Fan out independently and distinguish rejection from retryable failure."""
        results = await asyncio.gather(
            *(self.append_events(name, events) for name in task_names),
            return_exceptions=True,
        )
        deliveries, rejected, errors = {}, {}, {}
        for name, result in zip(task_names, results, strict=True):
            if isinstance(result, (TaskNotFoundError, TimeTriggeredTaskDeliveryError, InvalidInboxBatchError)):
                rejected[name] = str(result)
            elif isinstance(result, asyncio.CancelledError):
                errors[name] = "Delivery interrupted"
            elif isinstance(result, Exception):
                errors[name] = str(result)
            elif isinstance(result, BaseException):
                raise result
            else:
                deliveries[name] = result["event_ids"]
        return {"deliveries": deliveries, "rejected": rejected, "errors": errors}
