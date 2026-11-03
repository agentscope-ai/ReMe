"""Generate entity-oriented memory tags for added or modified Markdown files."""

from dataclasses import dataclass
import json
from typing import Literal

from agentscope.message import SystemMsg, UserMsg
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

    def __init__(
        self,
        direct_llm: bool = False,
        direct_excerpt_chars: int = 4500,
        direct_candidate_tags: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tools = ["read", "list_tags", "frontmatter_read", "frontmatter_update"]
        self.direct_llm = direct_llm
        self.direct_excerpt_chars = max(500, direct_excerpt_chars)
        self.direct_candidate_tags = max(0, direct_candidate_tags)

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

    @staticmethod
    def _document_excerpt(content: str, max_chars: int) -> str:
        """Keep the title-adjacent text and heading outline within a local model budget."""
        if len(content) <= max_chars:
            return content
        headings = [
            line.strip()
            for line in content.splitlines()
            if line.lstrip().startswith("#") and line.strip()
        ]
        heading_text = "\n".join(headings)
        if len(heading_text) > max_chars // 3:
            heading_text = heading_text[: max_chars // 3]
        body_budget = max(0, max_chars - len(heading_text) - 64)
        head = content[: body_budget // 2]
        tail = content[-(body_budget - len(head)) :] if body_budget > len(head) else ""
        return f"{head}\n\n[Headings]\n{heading_text}\n\n[Tail]\n{tail}"

    @staticmethod
    def _evidence_key(text: str) -> str:
        """Normalize text for conservative evidence checks."""
        return "".join(char.casefold() for char in text if char.isalnum())

    async def _direct_tags(self, target: _TagTarget) -> list[str]:
        """Generate tags in one structured call without an agent tool loop."""
        wrapper = self.agent_wrapper
        model = getattr(getattr(wrapper, "as_llm", None), "model", None)
        if model is None:
            raise RuntimeError("Direct auto-tag mode requires an agent wrapper with a bound LLM")
        file_store = self.file_store
        tag_index = file_store.require_tag_index()
        tag_key = tag_index.tag_key
        path = self.workspace_path / target.path
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
        metadata = post.metadata or {}

        candidates: list[str] = []
        if self.direct_candidate_tags:
            page = await tag_index.list_tags(
                order_by="file_count",
                order="desc",
                page_size=self.direct_candidate_tags,
            )
            candidates = [item[0] for item in page["items"]]

        excerpt = self._document_excerpt(post.content, self.direct_excerpt_chars)
        metadata_hint = {
            key: metadata[key]
            for key in ("name", "description", "kind")
            if key in metadata
        }
        messages = [
            SystemMsg(
                name="system",
                content=self.prompt_format(
                    "direct_system_prompt",
                    tag_key=tag_key,
                    max_tags_per_file=tag_index.max_tags_per_file,
                    existing_tags=", ".join(candidates) or "(none)",
                ),
            ),
            UserMsg(
                name="user",
                content=self.prompt_format(
                    "direct_user_message",
                    path=target.path,
                    metadata_json=json.dumps(metadata_hint, ensure_ascii=False, indent=2),
                    excerpt=excerpt,
                ),
            ),
        ]
        schema = {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "person",
                        "organization",
                        "product",
                        "place",
                        "asset",
                        "named_project",
                    ],
                },
                "evidence": {
                    "type": "string",
                    "description": "Exact short quote from the document body or description supporting the entity",
                },
                tag_key: {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": tag_index.max_tags_per_file,
                },
            },
            "required": ["entity_type", "evidence", tag_key],
            "additionalProperties": False,
        }
        response = await model.generate_structured_output(
            messages=messages,
            structured_model=schema,
        )
        content = response.content
        if hasattr(content, "model_dump"):
            content = content.model_dump()
        if not isinstance(content, dict):
            raise RuntimeError("Direct auto-tag model did not return an object")
        normalized = normalize_memory_tags(
            content.get(tag_key),
            max_tags_per_file=tag_index.max_tags_per_file,
            max_tag_length=tag_index.max_tag_length,
        )
        if str(content.get("entity_type") or "").strip().lower() == "none":
            return []
        evidence = str(content.get("evidence") or "").strip()
        body_without_headings = "\n".join(
            line for line in post.content.splitlines() if not line.lstrip().startswith("#")
        )
        searchable = "\n".join(
            value
            for value in (str(metadata.get("description") or ""), body_without_headings)
            if value
        )
        evidence_key = self._evidence_key(evidence)
        searchable_key = self._evidence_key(searchable)
        if not evidence_key or evidence_key not in searchable_key:
            self.logger.warning(f"[{self.name}] rejected unsupported tag evidence path={target.path}")
            return []
        supported = [tag for tag in normalized if self._evidence_key(tag) in evidence_key]
        if len(supported) != len(normalized):
            self.logger.warning(f"[{self.name}] rejected tags without evidence path={target.path}")
        return supported

    async def _process_target(self, target: _TagTarget) -> str:
        file_store = self.file_store
        tag_index = file_store.require_tag_index()
        tag_key = tag_index.tag_key
        tool_context = {
            "file_store": file_store.name,
            "_allowed_paths": [target.path],
            "_allowed_frontmatter_keys": [tag_key],
        }
        if self.direct_llm:
            generated_tags = await self._direct_tags(target)
            result_text = f"Generated tags: {json.dumps(generated_tags, ensure_ascii=False)}"
        else:
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
        if self.direct_llm:
            normalized = generated_tags
        elif tag_key not in metadata:
            raise RuntimeError(f"Auto-tag agent did not write the required {tag_key!r} field")
        else:
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
        return result_text if self.direct_llm else agent_reply_result_text(result)

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
