"""Atomic workspace controls shared by collectors and their lifecycle manager."""

import asyncio
from datetime import datetime, timezone

from ....schema.proactive_agent import validate_runtime_names
from ....schema.proactive_runtime import validate_agent_name
from ....utils.async_utils import complete_in_thread
from ....utils.files.path_lock import get_path_lock
from ....utils.files.workspace_files import WorkspaceFiles


def _read(files, path, name):
    value = files.read_json(path, None)
    if value is None:
        return {"schema_version": 1, "connector_job": name, "enabled": True}
    required = {"schema_version", "connector_job", "enabled", "updated_at"}
    if not isinstance(value, dict) or not required <= value.keys() <= required | {"runtime_names", "name"}:
        raise ValueError("Invalid persistent Connector control state")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["connector_job"] != name
        or type(value["enabled"]) is not bool
        or not isinstance(value["updated_at"], str)
    ):
        raise ValueError("Invalid persistent Connector control state")
    stamp = datetime.fromisoformat(value["updated_at"])
    if stamp.tzinfo is None:
        raise ValueError("Connector control time must include a timezone")
    if "runtime_names" in value:
        validate_runtime_names(value["runtime_names"], allow_empty=True)
    if "name" in value and (not isinstance(value["name"], str) or not 1 <= len(value["name"]) <= 80):
        raise ValueError("Invalid Connector name")
    return value


class ConnectorControls:
    """The original control files remain authoritative for preset collection gates."""

    def __init__(self, workspace, metadata_dir):
        self.files = WorkspaceFiles(workspace)
        self.directory = self.files.path(f"{metadata_dir}/proactive/connectors")

    def path(self, name):
        """Validate Job identities before joining paths."""
        return self.files.checked(self.directory / f"{validate_agent_name(name)}.json")

    async def read(self, name):
        """Absent overrides use the configured defaults."""
        return await asyncio.to_thread(_read, self.files, self.path(name), name)

    async def update(self, job_name, **changes):
        """Merge a partial patch under the same path lock used by every writer."""
        if not changes or changes.keys() - {"name", "enabled", "runtime_names"}:
            raise ValueError("Connector updates accept name, enabled and runtime_names")
        if "enabled" in changes and type(changes["enabled"]) is not bool:
            raise ValueError("Connector enabled must be a boolean")
        if "name" in changes and (not isinstance(changes["name"], str) or not 1 <= len(changes["name"]) <= 80):
            raise ValueError("Invalid Connector name")
        if "runtime_names" in changes:
            changes["runtime_names"] = validate_runtime_names(changes["runtime_names"], allow_empty=True)
        path = self.path(job_name)
        async with await get_path_lock(path):
            previous = await self.read(job_name)
            current = {**previous, **changes, "updated_at": datetime.now(timezone.utc).isoformat()}
            await complete_in_thread(self.files.write_json, path, current)
        return current
