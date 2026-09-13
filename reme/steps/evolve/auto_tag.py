"""Generate entity-oriented memory tags for added or modified Markdown files."""

from dataclasses import dataclass
from typing import Literal

import frontmatter

from ._evolve import agent_reply_result_text
from ..base_step import BaseStep
from ..file_io._path import display_path, resolve_path
from ..index import normalize_posix_path
from ...components import R
from ...constants import DEFAULT_MAX_MEMORY_TAG_LENGTH, DEFAULT_MAX_MEMORY_TAGS

_SUPPORTED_CHANGES = {"added", "modified"}


@dataclass(frozen=True)
class _TagTarget:
    change: Literal["added", "modified"]
    path: str


def normalize_memory_tags(
    value: object,
    *,
    max_tags_per_file: int = DEFAULT_MAX_MEMORY_TAGS,
    max_tag_length: int = DEFAULT_MAX_MEMORY_TAG_LENGTH,
) -> list[str]:
    """Normalize source tags while preserving canonical display casing."""
    if not isinstance(value, list):
        return []

    tags: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        tag = "_".join(item.split())
        if not tag or len(tag) > max_tag_length or not any(char.isalnum() for char in tag):
            continue
        canonical = tag.casefold()
        if canonical in seen:
            continue
        seen.add(canonical)
        tags.append(tag)
        if len(tags) >= max_tags_per_file:
            break
    return tags


@R.register("auto_tag_step")
class AutoTagStep(BaseStep):
    """Update memory tags for Markdown files described by the common ``changes`` contract."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tools = ["read", "list_tags", "frontmatter_read", "frontmatter_update"]

    def _targets(self) -> tuple[list[_TagTarget], list[dict[str, str]]]:
        """Validate, normalize, and de-duplicate added/modified Markdown changes."""
        assert self.context is not None
        raw_changes = self.context.get("changes") or []
        if not isinstance(raw_changes, list):
            raise ValueError("AutoTagStep requires changes: list[dict]")

        workspace = self.workspace_path.resolve()
        targets: dict[str, _TagTarget] = {}
        ignored: list[dict[str, str]] = []
        for item in raw_changes:
            if not isinstance(item, dict):
                ignored.append({"path": "", "reason": "change must be an object"})
                continue

            change = str(item.get("change") or "").strip().lower()
            raw_path = str(item.get("path") or "").strip()
            if change not in _SUPPORTED_CHANGES:
                ignored.append(
                    {
                        "path": raw_path,
                        "reason": f"unsupported change: {change or 'missing'}",
                    },
                )
                continue

            target, error = resolve_path(workspace, raw_path)
            if error or target is None:
                ignored.append({"path": raw_path, "reason": error or "invalid path"})
                continue
            if not target.is_file():
                ignored.append({"path": raw_path, "reason": "not a file"})
                continue
            if target.suffix.lower() != ".md":
                ignored.append({"path": raw_path, "reason": "not a Markdown file"})
                continue

            path = normalize_posix_path(display_path(workspace, target))
            previous = targets.get(path)
            was_added = previous is not None and previous.change == "added"
            normalized_change: Literal["added", "modified"] = "added" if change == "added" or was_added else "modified"
            targets[path] = _TagTarget(change=normalized_change, path=path)
        return list(targets.values()), ignored

    async def _process_target(self, target: _TagTarget) -> str:
        file_store = self.file_store
        tag_index = file_store.require_tag_index()
        tag_key = tag_index.tag_key
        tool_context = {
            "file_store": file_store.name,
            "_allowed_paths": [target.path],
            "_allowed_frontmatter_keys": [tag_key],
        }
        result = await self.agent_wrapper.reply(
            self.prompt_format("user_message", path=target.path, change=target.change, tag_key=tag_key),
            system_prompt=self.prompt_format(
                "system_prompt",
                tag_key=tag_key,
                max_tags_per_file=tag_index.max_tags_per_file,
            ),
            job_tools=self.tools,
            injected_job_kwargs=tool_context,
        )

        path = self.workspace_path / target.path
        metadata = dict(frontmatter.loads(path.read_text(encoding="utf-8")).metadata or {})
        normalized = normalize_memory_tags(
            metadata.get(tag_key),
            max_tags_per_file=tag_index.max_tags_per_file,
            max_tag_length=tag_index.max_tag_length,
        )
        if metadata.get(tag_key) != normalized:
            response = await self.run_job(
                "frontmatter_update",
                path=target.path,
                metadata={tag_key: normalized},
                **tool_context,
            )
            if not response.success:
                raise RuntimeError(str(response.answer))
        return agent_reply_result_text(result)

    async def execute(self):
        assert self.context is not None
        initial_success = self.context.response.success
        initial_answer = self.context.response.answer
        try:
            targets, ignored = self._targets()
        except ValueError as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        results: list[dict] = []
        for target in targets:
            try:
                summary = await self._process_target(target)
                results.append(
                    {
                        "change": target.change,
                        "path": target.path,
                        "success": True,
                        "summary": summary,
                    },
                )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.append(
                    {
                        "change": target.change,
                        "path": target.path,
                        "success": False,
                        "error": str(exc),
                    },
                )
                self.logger.warning(f"[{self.name}] failed path={target.path}: {exc}")

        failed = sum(not item["success"] for item in results)
        succeeded = len(results) - failed
        self.context.response.success = initial_success
        if not initial_answer and failed:
            self.context.response.answer = f"Tagged {succeeded} file(s); {failed} failed"
        elif not initial_answer and succeeded:
            self.context.response.answer = f"Tagged {succeeded} file(s)"
        else:
            self.context.response.answer = initial_answer
        self.context.response.metadata["auto_tag"] = {
            "processed": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "ignored": ignored,
            "results": results,
        }
        return self.context.response
