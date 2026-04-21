"""Lightweight file watcher for Markdown files only."""

from pathlib import Path

from watchfiles import Change

from .base_file_watcher import BaseFileWatcher
from ..component_registry import R


@R.register("light")
class LightFileWatcher(BaseFileWatcher):
    """Watches only Markdown files and delegates parsing to registered parsers."""

    async def _on_changes(self, changes: set[tuple[Change, str]]):
        """Handle file changes by filtering for Markdown files only."""
        if not self.file_store:
            self.logger.warning("File store not initialized, skipping changes")
            return

        # Filter changes to only include Markdown files
        md_changes = {
            (change_type, path) for change_type, path in changes if Path(path).suffix.lower() in [".md", ".markdown"]
        }

        for change_type, path in md_changes:
            try:
                if change_type in (Change.added, Change.modified):
                    # Use Markdown parser for Markdown files
                    parser = self._suffix_to_parser.get(".md", self._default_parser)
                    if not parser:
                        self.logger.warning(f"No parser available for Markdown file: {path}")
                        continue

                    file_meta, chunks = await parser.parse(path)
                    await self.file_store.upsert_file(file_meta, chunks)
                    self.logger.info(f"Upserted {len(chunks)} chunks for {file_meta.path}")

                elif change_type == Change.deleted:
                    await self.file_store.delete_file(path)
                    self.logger.info(f"Deleted {path}")

                else:
                    self.logger.warning(f"Unknown change type: {change_type}")

                self.logger.info(f"Markdown file {change_type} changed: {path}")

            except FileNotFoundError:
                self.logger.warning(f"File not found: {path}, skipping")
            except PermissionError:
                self.logger.warning(f"Permission denied: {path}, skipping")
            except Exception as e:
                self.logger.error(f"Error processing {path}: {e}", exc_info=True)
