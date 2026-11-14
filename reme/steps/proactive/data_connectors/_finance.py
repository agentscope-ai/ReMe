"""File-backed collection and retryable delivery through the existing append_event Job."""

import asyncio
from datetime import datetime
from hashlib import sha256
import math
from pathlib import Path

import httpx

from .base_connector import BaseConnectorStep
from ...file_io import check_path_permission, get_path_lock
from ....components.agent_runtime._internal.files import AgentFiles
from ....schema import Event, ResourceMetadata
from ....schema.agent_runtime import validate_agent_id
from ....utils.async_utils import complete_in_thread


class FinanceCollectorStep(BaseConnectorStep):
    """Keep upstream payloads and per-package delivery receipts in workspace files."""

    collected_at: datetime
    files: AgentFiles
    resource_root: Path

    def __init__(self, agent_id="finance", max_retries=3, request_timeout=20, request_interval=1, **kwargs):
        super().__init__(**kwargs)
        self.agent_id = validate_agent_id(agent_id)
        if type(max_retries) is not int or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        for name, value in (("request_timeout", request_timeout), ("request_interval", request_interval)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if request_timeout == 0:
            raise ValueError("request_timeout must be positive")
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.request_interval = request_interval

    def resource_target(self):
        target, self.collected_at = super().resource_target()
        return target, self.collected_at

    async def execute(self):
        target, _ = self.resource_target()
        self.files = AgentFiles(self.workspace_path)
        self.resource_root = self.files.checked(target.parent.parent)
        self.require_path(self.resource_root)
        self.require_path(self.files.path(self.config_value("reme_agent_dir")))
        if self.get_job("append_event") is None:
            raise RuntimeError("Finance collectors require the append_event Job")
        async with await get_path_lock(self.resource_root):
            await self._deliver_pending()
            await super().execute()
            await self._deliver_pending()
        return self.context.response

    def require_path(self, path: Path):
        """Authorize history and receipt access before network or file I/O."""
        self.files.checked(path)
        if not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")):
            raise PermissionError("no permission to access finance collector files")

    def manifests(self):
        """Published manifests are recoverable collection evidence, including unsent batches."""
        for path in sorted(self.resource_root.glob("*/*/metadata.json")):
            self.require_path(path)
            yield path

    def _already_collected(self):
        """Use published packages to avoid duplicate quote snapshots in the same minute."""
        minute = self.collected_at.replace(second=0, microsecond=0)
        day = self.collected_at.date().isoformat()
        for manifest in self.manifests():
            if manifest.parent.parent.name != day:
                continue
            metadata = ResourceMetadata.model_validate_json(self.files.read(manifest))
            if metadata.created_at.replace(second=0, microsecond=0) == minute:
                return True
        return False

    async def _deliver_pending(self):
        manifests = await asyncio.to_thread(lambda: list(self.manifests()))
        for path in manifests:
            receipt = path.parent / f"delivery-{self.agent_id}.json"
            self.require_path(receipt)
            relative = path.parent.relative_to(self.workspace_path).as_posix()
            event_id = f"resource-{sha256(relative.encode()).hexdigest()}"
            expected = {"agent_id": self.agent_id, "event_id": event_id}
            saved = await asyncio.to_thread(self.files.read_json, receipt, None)
            if saved is not None:
                if saved != expected:
                    raise ValueError(f"Invalid finance delivery receipt: {receipt}")
                continue
            metadata = ResourceMetadata.model_validate_json(await asyncio.to_thread(self.files.read, path))
            if metadata.source != self.source:
                raise ValueError(f"Unexpected finance source: {metadata.source}")
            payloads = [path, *(path.parent / item.path for item in metadata.files)]
            for payload in payloads:
                self.require_path(payload)
                if not payload.is_file():
                    raise ValueError(f"Missing finance resource: {payload}")
            event = Event(
                id=event_id,
                event_ts=metadata.created_at,
                source=self.source,
                description=metadata.description,
                paths=[item.relative_to(self.workspace_path).as_posix() for item in payloads],
            )
            response = await self.run_job(
                "append_event",
                agent_id=self.agent_id,
                events=[event.model_dump(mode="json")],
                _allowed_paths=self.context.get("_allowed_paths"),
            )
            if not response.success:
                raise RuntimeError(f"Finance event delivery failed: {response.answer}")
            await complete_in_thread(self.files.write_json, receipt, expected)

    def check_stopped(self):
        """Interrupt pagination and retry delays when the owning Job is stopped."""
        if self._cancelled.is_set() or (self.context.stop_event is not None and self.context.stop_event.is_set()):
            raise RuntimeError("Finance collection stopped")

    def wait(self, seconds):
        """Wait off-loop in short, cancellable intervals."""
        remaining = seconds
        while remaining > 0:
            self.check_stopped()
            delay = min(remaining, 0.1)
            self._cancelled.wait(delay)
            remaining -= delay
        self.check_stopped()

    def request(self, client: httpx.Client, url: str, *, params=None) -> bytes:
        """Bound each HTTP response and retry transient transport or HTTP failures."""
        for attempt in range(self.max_retries):
            self.check_stopped()
            try:
                with client.stream("GET", url, params=params) as response:
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_bytes():
                        self.check_stopped()
                        data.extend(chunk)
                        if len(data) > self.max_file_bytes():
                            raise ValueError("Finance response exceeds max_file_bytes")
                    return bytes(data)
            except httpx.HTTPError:
                if attempt + 1 == self.max_retries:
                    raise
                self.wait(2**attempt)
        raise RuntimeError("Finance request exhausted retries")

    def write_records(self, directory: Path, filename: str, records: list) -> list[dict]:
        """Publish nonempty records only, within the existing per-file byte limit."""
        if not records:
            return []
        content = "".join(record.model_dump_json() + "\n" for record in records)
        if len(content.encode("utf-8")) > self.max_file_bytes():
            raise ValueError("Finance collection exceeds max_file_bytes")
        self.check_stopped()
        (directory / filename).write_text(content, encoding="utf-8")
        return [{"path": filename}]
