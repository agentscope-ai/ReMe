"""File-backed collection and retryable delivery to named runtimes."""

from datetime import datetime
import math
from pathlib import Path

import httpx

from .base_connector import BaseConnectorStep
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ResourceMetadata


class FinanceCollectorStep(BaseConnectorStep):
    """Keep upstream payloads and per-package delivery receipts in workspace files."""

    collected_at: datetime
    files: WorkspaceFiles
    resource_root: Path

    def __init__(self, task_names=None, max_retries=3, request_timeout=20, request_interval=1, **kwargs):
        super().__init__(task_names=["finance"] if task_names is None else task_names, **kwargs)
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
        self.files = WorkspaceFiles(self.workspace_path)
        self.resource_root = self.files.checked(target.parent.parent)
        return target, self.collected_at

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

    def saved_records(self, filename, model):
        """Read retained source records; derived quote state can always be rebuilt."""
        for manifest in self.manifests():
            path = manifest.parent / filename
            self.require_path(path)
            for value in self.files.iter_jsonl(path):
                self.check_stopped()
                yield model.model_validate(value)

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
