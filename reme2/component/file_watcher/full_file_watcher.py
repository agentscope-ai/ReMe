"""Full file watcher for all supported file types."""

from pathlib import Path

from watchfiles import Change

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R


@R.register("full")
class FullFileWatcher(BaseFileWatcher):
    """Watches all supported file types and delegates parsing to registered parsers."""

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes by delegating to appropriate parsers."""
        if not self.file_store:
            self.logger.warning("File store not initialized, skipping changes")
            return

        for change_type, path in changes:
            try:
                if change_type in (Change.added, Change.modified):
                    suffix = Path(path).suffix.lower()
                    parser = self._suffix_to_parser.get(suffix, self._default_parser)
                    if not parser:
                        self.logger.debug(f"No parser available for file type {suffix}: {path}, skipping")
                        continue

                    file_meta, chunks = await parser.parse(path)
                    await self.file_store.upsert_file(file_meta, chunks)
                    self.logger.info(f"Upserted {len(chunks)} chunks for {file_meta.path}")

                elif change_type == Change.deleted:
                    await self.file_store.delete_file(path)
                    self.logger.info(f"Deleted {path}")

                else:
                    self.logger.warning(f"Unknown change type: {change_type}")

                self.logger.info(f"File {change_type} changed: {path}")

            except FileNotFoundError:
                self.logger.warning(f"File not found: {path}, skipping")
            except PermissionError:
                self.logger.warning(f"Permission denied: {path}, skipping")
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}", exc_info=True)
