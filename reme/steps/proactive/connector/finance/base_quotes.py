"""Shared quote polling, source validation and snapshot deduplication."""

from abc import abstractmethod
from datetime import datetime
from pathlib import Path

import httpx

from .base_finance import FinanceCollectorStep
from .....schema import ResourceMetadata
from .....schema.resource_metadata import RESOURCE_METADATA_FILENAME


class QuoteCollectorStep(FinanceCollectorStep):
    """Share polling and deduplication across independent quote feeds."""

    filename: str
    quote_model: type

    def _already_collected(self):
        """Use published packages to avoid duplicate quote snapshots in the same minute."""
        minute = self.collected_at.replace(second=0, microsecond=0)
        pattern = f"{minute:%Y-%m-%d}/{minute:%H%M}??_*/{RESOURCE_METADATA_FILENAME}"
        for manifest in self.resource_root.glob(pattern):
            self.require_paths(manifest)
            metadata = ResourceMetadata.model_validate_json(self.files.read(manifest))
            if metadata.created_at.replace(second=0, microsecond=0) == minute:
                return True
        return False

    def _saved_quotes(self):
        """Rebuild quote state from published resources."""
        for manifest in self.manifests():
            path = manifest.parent / self.filename
            self.require_paths(path)
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
