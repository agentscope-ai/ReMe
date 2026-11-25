"""Shared HTTP collection for financial news and quotes."""

from abc import abstractmethod
from datetime import datetime
import math
from pathlib import Path

import httpx

from .base_connector import BaseConnectorStep
from ....utils.files.workspace_files import WorkspaceFiles
from ....schema import ResourceMetadata
from ....schema.resource_metadata import RESOURCE_METADATA_FILENAME


class FinanceCollectorStep(BaseConnectorStep):
    """Read and publish financial source records within the resource package lifecycle."""

    collected_at: datetime
    files: WorkspaceFiles
    resource_root: Path

    def __init__(self, task_names=None, max_retries=3, request_timeout=20, **kwargs):
        super().__init__(task_names=["proactive_fin"] if task_names is None else task_names, **kwargs)
        if type(max_retries) is not int or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        if type(request_timeout) not in (int, float) or not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def resource_target(self):
        target, self.collected_at = super().resource_target()
        self.files = WorkspaceFiles(self.workspace_path)
        self.resource_root = self.files.checked(target.parent.parent)
        return target, self.collected_at

    def manifests(self):
        """Find published source packages, including pending deliveries."""
        for path in self.resource_root.glob(f"*/*/{RESOURCE_METADATA_FILENAME}"):
            self.require_path(path)
            yield path

    def client(self, **kwargs):
        """Use the same timeouts across finance feeds."""
        return httpx.Client(timeout=httpx.Timeout(self.request_timeout, connect=5), **kwargs)

    def write_records(self, directory, filename, records):
        """Write a bounded, nonempty JSONL payload."""
        if not records:
            return []
        content = "".join(record.model_dump_json() + "\n" for record in records)
        if len(content.encode("utf-8")) > self.max_file_bytes():
            raise ValueError("Finance data exceeds max_file_bytes")
        self.check_stopped()
        (directory / filename).write_text(content, encoding="utf-8")
        return [{"path": filename}]

    def check_stopped(self):
        """Interrupt HTTP work and retry delays when the owning Job is stopped."""
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


class QuoteCollectorStep(FinanceCollectorStep):
    """Share polling and deduplication across independent quote feeds."""

    filename: str
    quote_model: type

    def _already_collected(self):
        """Use published packages to avoid duplicate quote snapshots in the same minute."""
        minute = self.collected_at.replace(second=0, microsecond=0)
        pattern = f"{minute:%Y-%m-%d}/{minute:%H%M}??_*/{RESOURCE_METADATA_FILENAME}"
        for manifest in self.resource_root.glob(pattern):
            self.require_path(manifest)
            metadata = ResourceMetadata.model_validate_json(self.files.read(manifest))
            if metadata.created_at.replace(second=0, microsecond=0) == minute:
                return True
        return False

    def _saved_quotes(self):
        """Rebuild quote state from published resources."""
        for manifest in self.manifests():
            path = manifest.parent / self.filename
            self.require_path(path)
            for value in self.files.iter_jsonl(path):
                self.check_stopped()
                yield self.quote_model.model_validate(value)

    @abstractmethod
    def fetch_quote(self, client: httpx.Client):
        """Read and validate the provider's quote."""

    @staticmethod
    @abstractmethod
    def changed(quote, previous):
        """Check whether the quote is new and materially different."""

    def received_at(self):
        """Use the collection timezone for the time the response was received."""
        return datetime.now(self.collected_at.tzinfo)

    def collect(self, directory: Path) -> list[dict]:
        if self._already_collected():
            return []
        with self.client() as client:
            quote = self.fetch_quote(client)
        previous = max(self._saved_quotes(), key=lambda item: item.collected_at, default=None)
        if previous is not None and not self.changed(quote, previous):
            return []
        return self.write_records(directory, self.filename, [quote])
