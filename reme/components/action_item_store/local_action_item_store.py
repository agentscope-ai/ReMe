"""Local action-item store: JSONL in the workspace, reloaded into memory."""

import asyncio
import json
import os

from pydantic import ValidationError

from .base_action_item_store import BaseActionItemStore
from ..component_registry import R
from ...schema import ActionItem, ActionItemQuery, apply_update


@R.register("local")
class LocalActionItemStore(BaseActionItemStore):
    """Keep one JSONL file of action items under the workspace metadata directory.

    The file is the durable record: every restart reloads it, and the in-memory
    map is only an index that can always be rebuilt from the file. A corrupt
    line is skipped with a warning so one bad record cannot hide the rest.
    """

    def __init__(self, file_name: str = "action_items.jsonl", **kwargs):
        super().__init__(**kwargs)
        self.file_name = file_name
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        self._file = self.component_metadata_path / file_name
        self._items: dict[str, ActionItem] = {}
        self._io_lock = asyncio.Lock()

    async def load(self) -> None:
        async with self._io_lock:
            if not self._file.exists():
                return
            loaded: dict[str, ActionItem] = {}
            with self._file.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = ActionItem.model_validate(json.loads(text))
                    except (json.JSONDecodeError, ValidationError) as exc:
                        self.logger.warning(f"Skipped invalid action item at {self._file}:{number}: {exc}")
                        continue
                    loaded[item.id] = item
            self._items = loaded
            self.logger.debug(f"Loaded {len(loaded)} action items from {self._file}")

    async def dump(self) -> None:
        async with self._io_lock:
            self._write()
            self.logger.info(f"Saved {len(self._items)} action items to {self._file}")

    def _write(self) -> None:
        self.component_metadata_path.mkdir(parents=True, exist_ok=True)
        temporary = self._file.with_name(f"{self._file.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for item in sorted(self._items.values(), key=lambda record: record.id):
                handle.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
                handle.write("\n")
        os.replace(temporary, self._file)

    async def upsert(self, items: list[ActionItem]) -> list[ActionItem]:
        async with self._io_lock:
            stored: list[ActionItem] = []
            for incoming in items:
                existing = self._items.get(incoming.id)
                merged = incoming if existing is None else apply_update(existing, incoming)
                self._items[merged.id] = merged
                stored.append(merged)
            if stored:
                self._write()
            return stored

    async def get(self, item_id: str) -> ActionItem | None:
        async with self._io_lock:
            return self._items.get(item_id)

    async def query(self, query: ActionItemQuery | None = None) -> list[ActionItem]:
        filter_ = query or ActionItemQuery()
        async with self._io_lock:
            matched = [item for item in self._items.values() if filter_.matches(item)]
        matched.sort(key=lambda item: item.sort_key(), reverse=True)
        return matched[: filter_.limit]
