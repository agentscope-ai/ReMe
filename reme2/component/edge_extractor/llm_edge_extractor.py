"""LLMEdgeExtractor — slow route per `structure.md` §"双态抽取".

ReAct agent over the memory store: instead of a single-shot extraction
from the file's text alone, the agent browses the vault via read-only
memory tools (`memory_get` / `memory_list` / `memory_links` /
`memory_backlinks` / `memory_resolve_wikilink`) to ground its triples
in entities that already exist. It finalizes by calling `emit_edges`
with the JSON triples it wants to commit.

Each accepted triple becomes a `FileEdge` with `source="llm"` and
`confidence` populated from the model's self-rating.

The extractor does NOT write back to the source file — that's the
caller's job (typically the parser pipeline). It also does NOT
entity-resolve targets; the raw `object` string from the model lands
in `FileEdge.target` and is matched at the file_store layer via the
usual wikilink resolution pipeline.

Inputs that exceed `max_input_chars` are truncated; long files
should be chunked by the caller before invoking this extractor.

Triples are filtered by `min_confidence` and predicate-normalized
(lowercase, non-identifier characters → underscore) when
`predicate_normalizer=True`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import frontmatter
from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit, ToolResponse
from pydantic import BaseModel, Field

from .base_edge_extractor import BaseEdgeExtractor
from ..component_registry import R
from ..file_store import BaseFileStore
from ...enumeration import ComponentEnum
from ...schema import FileEdge


_SYSTEM_PROMPT = """You are a high-precision information-extraction agent for a personal markdown vault.

Your job for one file: identify subject-predicate-object triples that capture meaningful, lasting relations between named entities, then emit them by calling `emit_edges`. Quality > recall.

You have read-only access to the vault via these tools:
  - `memory_list(path_prefix, tags, limit)`        — browse indexed files
  - `memory_get(path)`                              — read frontmatter + body + edges of one file
  - `memory_resolve_wikilink(wikilink)`             — check whether `[[X]]` resolves to a real path
  - `memory_links(path)`                            — outgoing edges from `path`
  - `memory_backlinks(path)`                        — incoming edges to `path`
  - `emit_edges(triples)`                           — finalize: list of {subject, predicate, object, confidence}

Recommended loop (a few iterations is fine; do NOT exhaust the vault):
  1. Read the source file's text and metadata (provided in the user message).
  2. Generate candidate object entities from the text.
  3. Probe the vault: prefer `memory_resolve_wikilink` to confirm a candidate exists; fall back to `memory_list` for fuzzy browsing only when needed.
  4. Drop candidates that are pronouns / generic concepts / vague references / self-references.
  5. Call `emit_edges` ONCE with the final filtered set, then stop.

Triple rules:
  - subject: the source file's main entity (typically the file name / first H1).
  - predicate: identifier-shaped, lowercase, words joined with underscore (e.g. `works_at`, `authored`, `depends_on`, `mentions`).
  - object: the target entity name as it appears in the text — do NOT wrap with `[[ ]]`.
  - confidence: float in [0, 1]; the parser drops anything below `min_confidence`.

Skip silently: pronouns, self-referential relations, transient events, generic concepts.
If nothing meaningful is present, call `emit_edges([])` and stop."""


_USER_TEMPLATE = """Source file: {path}

Frontmatter:
{frontmatter}

Body:
{text}

Extract triples grounded in the vault now. Use the read tools as needed, then call `emit_edges` ONCE with the final list."""


class _Triple(BaseModel):
    """One subject-predicate-object triple from the LLM."""

    subject: str = Field(description="Source entity of the relation.")
    predicate: str = Field(description="Identifier-shaped relation name.")
    object: str = Field(description="Target entity name.")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-rated confidence.")


_LINK_WRAPPER_RE = re.compile(r"^!?\[\[([^\]\|\#]+?)(?:[#\|][^\]]*)?\]\]$")
_PREDICATE_NORM_RE = re.compile(r"[^a-z0-9_]+")


def _strip_link_wrapper(s: str) -> str:
    """Strip `[[X]]` / `![[X#h|alias]]` wrappers if the LLM emitted them."""
    s = s.strip().strip('"\'')
    if m := _LINK_WRAPPER_RE.match(s):
        return m.group(1).strip()
    return s


def _normalize_predicate(pred: str) -> str:
    """Lowercase + collapse non-identifier chars to `_`."""
    pred = pred.strip().lower()
    pred = _PREDICATE_NORM_RE.sub("_", pred).strip("_")
    return pred or "related"


def _text_response(payload: object) -> ToolResponse:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return ToolResponse(content=[TextBlock(type="text", text=text)])


@R.register("llm")
class LLMEdgeExtractor(BaseEdgeExtractor):
    """LLM-driven typed-edge extractor (slow route, ReAct over memory tools)."""

    def __init__(
        self,
        as_llm: str = "default",
        as_llm_formatter: str = "default",
        file_store: str = "default",
        max_input_chars: int = 8000,
        min_confidence: float = 0.5,
        predicate_normalizer: bool = True,
        max_iters: int = 8,
        console_enabled: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._as_llm_name: str = as_llm
        self._as_llm_formatter_name: str = as_llm_formatter
        self._file_store_name: str = file_store
        self.max_input_chars: int = int(max_input_chars)
        self.min_confidence: float = float(min_confidence)
        self.predicate_normalizer: bool = bool(predicate_normalizer)
        self.max_iters: int = int(max_iters)
        self.console_enabled: bool = bool(console_enabled)
        self.as_llm: ChatModelBase | None = None
        self.as_llm_formatter: FormatterBase | None = None
        self.file_store: BaseFileStore | None = None

    async def _start(self) -> None:
        assert self.app_context is not None, "LLMEdgeExtractor requires app_context"

        llms = self.app_context.components.get(ComponentEnum.AS_LLM, {})
        wrapper = llms.get(self._as_llm_name)
        if wrapper is None:
            raise ValueError(f"as_llm '{self._as_llm_name}' not configured")
        model = getattr(wrapper, "model", None)
        if not isinstance(model, ChatModelBase):
            raise TypeError(
                f"as_llm '{self._as_llm_name}'.model is {type(model).__name__}, "
                f"expected ChatModelBase",
            )
        self.as_llm = model

        formatters = self.app_context.components.get(ComponentEnum.AS_LLM_FORMATTER, {})
        fwrapper = formatters.get(self._as_llm_formatter_name)
        formatter = getattr(fwrapper, "formatter", None) if fwrapper is not None else None
        if not isinstance(formatter, FormatterBase):
            raise TypeError(
                f"as_llm_formatter '{self._as_llm_formatter_name}' missing or wrong type",
            )
        self.as_llm_formatter = formatter

        stores = self.app_context.components.get(ComponentEnum.FILE_STORE, {})
        store = stores.get(self._file_store_name)
        if not isinstance(store, BaseFileStore):
            raise ValueError(
                f"file_store '{self._file_store_name}' not configured for LLMEdgeExtractor",
            )
        self.file_store = store

    async def _close(self) -> None:
        self.as_llm = None
        self.as_llm_formatter = None
        self.file_store = None

    async def extract(
        self,
        text: str,
        metadata: dict | None = None,
        path: str | None = None,
    ) -> list[FileEdge]:
        if not text or not text.strip():
            return []
        if self.as_llm is None or self.as_llm_formatter is None or self.file_store is None:
            raise RuntimeError("LLMEdgeExtractor not started; call .start() first")

        body = text if len(text) <= self.max_input_chars else text[: self.max_input_chars]

        toolkit, sink = self._build_toolkit()

        agent = ReActAgent(
            name="llm_edge_extractor",
            model=self.as_llm,
            sys_prompt=_SYSTEM_PROMPT,
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
            max_iters=self.max_iters,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_msg = _USER_TEMPLATE.format(
            path=path or "(unknown)",
            frontmatter=json.dumps(metadata or {}, ensure_ascii=False, indent=2, default=str),
            text=body,
        )

        try:
            await agent.reply(Msg(name="user", role="user", content=user_msg))
        except Exception as e:
            self.logger.warning(f"LLM IE agent failed for {path}: {e}")
            return []

        return self._sink_to_edges(sink)

    # -- Tool plumbing ------------------------------------------------------

    def _build_toolkit(self) -> tuple[Toolkit, list[_Triple]]:
        """Build a read-only memory toolkit + an `emit_edges` sink.

        Returns `(toolkit, sink)`. The sink is the mutable list the
        `emit_edges` tool appends parsed triples into.
        """
        assert self.file_store is not None
        store = self.file_store
        vault_root = store.vault_root or Path.cwd().resolve()

        sink: list[_Triple] = []

        def _resolve(p: str) -> Path:
            pp = Path(p)
            if not pp.is_absolute():
                pp = vault_root / pp
            return pp.resolve()

        async def memory_get(path: str) -> ToolResponse:
            """Read one vault file: frontmatter + body + outgoing edges.

            Args:
                path (str): Absolute or vault-relative path to the file.
            """
            target = _resolve(path)
            out: dict = {"path": str(target), "exists": False}
            meta = store.get_file_meta(str(target))
            if meta is not None:
                edges = store.get_edges(str(target))
                out.update({
                    "exists": True,
                    "metadata": meta.metadata,
                    "edges": [e.model_dump(exclude_none=True) for e in edges],
                })
            if target.is_file():
                try:
                    raw = target.read_text(encoding="utf-8")
                    post = frontmatter.loads(raw)
                    out["exists"] = True
                    out["metadata"] = dict(post.metadata)
                    out["content"] = post.content
                except Exception as e:
                    out["read_error"] = str(e)
            return _text_response(out)

        async def memory_list(
            path_prefix: str = "",
            tags: list[str] | None = None,
            limit: int = 50,
        ) -> ToolResponse:
            """List indexed vault files, filtered by path prefix and tags.

            Args:
                path_prefix (str): Restrict to paths starting with this prefix.
                tags (list[str] | None): All tags must be present on a file.
                limit (int): Cap on returned items.
            """
            items: list[dict] = []
            for p, meta in store.nodes.items():
                if path_prefix and not p.startswith(path_prefix):
                    continue
                md = meta.metadata or {}
                if tags:
                    file_tags = set(md.get("tags", []) or [])
                    if not all(t in file_tags for t in tags):
                        continue
                items.append({"path": p, "file": meta.file, "metadata": md})
                if len(items) >= limit:
                    break
            return _text_response({"items": items, "count": len(items)})

        async def memory_resolve_wikilink(wikilink: str) -> ToolResponse:
            """Resolve a `[[wikilink]]` to its absolute path (or null if dangling).

            Args:
                wikilink (str): The bare wikilink target (no brackets).
            """
            hit = store.resolve_wikilink(wikilink)
            return _text_response({"wikilink": wikilink, "path": hit})

        async def memory_links(path: str) -> ToolResponse:
            """Outgoing edges from `path` (resolved targets only).

            Args:
                path (str): Absolute or vault-relative path.
            """
            target = _resolve(path)
            out = [
                {"path": m.path, "predicate": e.predicate, "metadata": m.metadata}
                for m, e in store.get_links(str(target))
            ]
            return _text_response({"path": str(target), "links": out})

        async def memory_backlinks(path: str) -> ToolResponse:
            """Incoming edges to `path` (files that link TO it).

            Args:
                path (str): Absolute or vault-relative path.
            """
            target = _resolve(path)
            out = [
                {"path": m.path, "predicate": e.predicate, "metadata": m.metadata}
                for m, e in store.get_backlinks(str(target))
            ]
            return _text_response({"path": str(target), "backlinks": out})

        async def emit_edges(triples: list[dict]) -> ToolResponse:
            """Finalize extraction. Emit the full set of triples for this file.

            Call this exactly ONCE per file, with all triples you want to
            commit. After this call, stop and produce a brief textual reply
            summarizing what you emitted.

            Args:
                triples (list[dict]): List of {subject, predicate, object, confidence}.
                    `confidence` ∈ [0, 1]; values below the configured threshold
                    are dropped by the parser.
            """
            accepted = 0
            rejected = 0
            for t in triples:
                try:
                    sink.append(_Triple.model_validate(t))
                    accepted += 1
                except Exception:
                    rejected += 1
            return _text_response({
                "accepted": accepted,
                "rejected": rejected,
                "total_so_far": len(sink),
            })

        toolkit = Toolkit()
        for fn in (
            memory_get,
            memory_list,
            memory_resolve_wikilink,
            memory_links,
            memory_backlinks,
            emit_edges,
        ):
            toolkit.register_tool_function(fn, namesake_strategy="override")
        return toolkit, sink

    def _sink_to_edges(self, sink: list[_Triple]) -> list[FileEdge]:
        edges: list[FileEdge] = []
        for tri in sink:
            if tri.confidence < self.min_confidence:
                continue
            target = _strip_link_wrapper(tri.object)
            if not target or target == _strip_link_wrapper(tri.subject):
                continue
            predicate = (
                _normalize_predicate(tri.predicate)
                if self.predicate_normalizer
                else tri.predicate.strip()
            )
            edges.append(FileEdge(
                target=target,
                predicate=predicate or None,
                source="llm",
                confidence=tri.confidence,
            ))
        return edges
