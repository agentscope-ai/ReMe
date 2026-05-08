"""topic_create — create a new topic under topics/{folder}/{name}.md.

The only file-creation entry point for topics. Enforces:
    - path template (topics/{folder}/{name}.md)
    - folder topic identification (folder == name)
    - judgment-category strong-confidence (via Topic.model_validator)
    - wikilink uniqueness (creating this file must not make `[[name]]` ambiguous)

Writes via `write_create` — the canonical L1 invariant gate that also
performs the wikilink-uniqueness check. The Ingestor's R-M-W loop is
reserved for content-driven flows; topic creation is path-driven, so
the direct call is sufficient.
"""

import json
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from reme2.component import R
from reme2.component.base_step import BaseStep
from reme2.memory.memory_io import write_create
from reme2.schema.vault import Topic
from reme2.utils.vault_paths import next_suffixed_stem, topic_path


@R.register("topic_create")
class TopicCreate(BaseStep):
    """Create a topic file with the standard frontmatter template."""

    def __init__(self, vault_root: str = "", topics_dir: str = "topics", **kwargs):
        super().__init__(**kwargs)
        self.vault_root = vault_root
        self.topics_dir = topics_dir

    def _root(self) -> Path:
        if self.vault_root:
            return Path(self.vault_root)
        watcher = self.app_context.components["file_watcher"]["default"]  # type: ignore[union-attr]
        return Path(watcher.watch_path)

    async def execute(self):
        assert self.context is not None
        folder: str = self.context.get("folder", "")
        name: str = self.context.get("name", "")
        category: str = self.context.get("category", "")
        description: str = self.context.get("description", "") or ""
        content: str = self.context.get("content", "") or ""
        confidence = self.context.get("confidence")
        market = self.context.get("market")
        ticker = self.context.get("ticker")
        tags: list[str] = self.context.get("tags") or []

        assert folder, "folder is required"
        assert name, "name is required"
        assert category, "category is required"

        target = topic_path(self._root(), folder, name, self.topics_dir)
        if target.exists():
            self.context.response.success = False
            self.context.response.answer = json.dumps({
                "path": str(target.resolve()),
                "error": "topic already exists; use memory_update / memory_property_update to modify",
            }, ensure_ascii=False)
            return

        today = date.today().isoformat()
        metadata: dict = {
            "title": name,
            "description": description,
            "category": category,
            "tags": tags,
            "created": today,
            "updated": today,
        }
        if confidence is not None:
            metadata["confidence"] = confidence
        if market is not None:
            metadata["market"] = market
        if ticker is not None:
            metadata["ticker"] = ticker

        # Topic schema validation (judgment categories require confidence).
        try:
            Topic.model_validate(metadata)
        except ValidationError as e:
            self.context.response.success = False
            self.context.response.answer = json.dumps({
                "error": "Topic schema validation failed",
                "details": e.errors(include_context=False, include_url=False),
            }, ensure_ascii=False)
            return

        # Pre-check uniqueness so we can surface a `suggested_name` to the
        # agent. The actual write also routes through the same gate inside
        # write_create, so this is a UX nicety, not a correctness check.
        graph = self.file_store
        conflicts = graph.collisions_after_create(target)
        if conflicts:
            taken = {Path(p).stem for p in graph.nodes}
            suggested_name = next_suffixed_stem(taken, name)
            self.context.response.success = False
            self.context.response.answer = json.dumps({
                "error": (
                    f"stem `[[{name}]]` would resolve ambiguously "
                    f"to {len(conflicts) + 1} paths after this create"
                ),
                "conflicts": conflicts,
                "suggested_name": suggested_name,
                "hint": (
                    f"retry with name='{suggested_name}' (numeric suffix), "
                    f"OR use a domain-specific qualifier (e.g. '{name}-Inc' / "
                    f"'{name}-v2') — semantic names beat numeric. If you "
                    f"actually meant the existing topic, call memory_get on "
                    f"one of `conflicts` instead."
                ),
            }, ensure_ascii=False)
            return

        ok, payload = write_create(
            self.file_store, target,
            metadata=metadata, content=content,
        )

        if not ok:
            self.context.response.success = False
            self.context.response.answer = json.dumps({
                "path": str(target.resolve()),
                "error": payload.get("error", "create failed"),
                "details": payload,
            }, ensure_ascii=False)
            return

        is_folder_topic = folder == name
        self.context.response.success = True
        self.context.response.answer = json.dumps({
            "path": str(target.resolve()),
            "category": category,
            "is_folder_topic": is_folder_topic,
            "created": True,
        }, ensure_ascii=False)
