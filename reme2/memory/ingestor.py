"""Smart Ingestor — ReAct agent over the Memory File System.

Per `structure.md` L31-44, the Ingestor is the SSOT engine: the **single
write entry point** to the markdown vault. Every mutation (create, body
edit, frontmatter flip, rename, delete, archive) flows through here.

Mirrors `Summarizer`'s pattern — drives a `ReActAgent` whose toolkit is
built by `memory_toolkit.build_memory_toolkit`. The agent runs its own
R-M-W loop: read related files via tools, decide which ones to mutate,
call the right write tool. Every write tool records into an audit list,
so the caller gets a deterministic mutation trail regardless of how the
agent's reasoning unfolded.

When no LLM is configured, falls back to a direct create from
`target_path` + `metadata` + `content`. Edits/renames/deletes are not
available without an LLM.
"""

from __future__ import annotations

import datetime
import json
import zoneinfo
from pathlib import Path

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ..component.runtime_response import _set_answer, _to_jsonable
from . import memory_io
from .memory_io import create_file
from .memory_toolkit import build_memory_toolkit
from ..component import R
from ..component.base_step import BaseStep
from ..enumeration import ComponentEnum
from ..utils.wikilink import extract_wikilinks


class IngestResult(BaseModel):
    """Audit trail for a single ingest call."""

    applied: list[dict] = Field(default_factory=list, description="Successful ops with paths.")
    rejected: list[dict] = Field(default_factory=list, description="Ops the validator refused.")
    failed: list[dict] = Field(default_factory=list, description="Ops that errored at apply time.")
    skipped: bool = Field(default=False, description="True if the LLM returned a SkipOp.")
    used_llm: bool = Field(default=False, description="False = degraded path (no LLM configured).")

    @property
    def success(self) -> bool:
        """Skipped or any-applied with no failures = success."""
        if self.skipped:
            return True
        return len(self.applied) > 0 and len(self.failed) == 0


@R.register("ingestor")
class Ingestor(BaseStep):
    """R-M-W ingestor exposed as a single-write-entry MCP tool.

    Inputs (read from RuntimeContext):
        content      (str, required): the material being ingested.
        hint         (str, optional): caller guidance to the LLM.
        target_path  (str, optional): suggested file path; required for
            the degraded path (no LLM).
        metadata     (dict, optional): suggested frontmatter; used by
            the degraded path or as a hint to the LLM.
        related_paths (list[str], optional): explicit related files,
            auto-extended with wikilinks parsed from `content`.

    Output (written to context.response.answer):
        IngestResult JSON — applied/failed lists plus used_llm flag and
        the agent's final-message summary.
    """

    def __init__(
        self,
        toolkit: Toolkit | None = None,
        console_enabled: bool = False,
        timezone: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.toolkit = toolkit
        self.console_enabled = console_enabled
        self.timezone = timezone
        self._protocol = (Path(__file__).parent / "protocol.md").read_text(encoding="utf-8")

    def _now(self) -> datetime.datetime:
        if self.timezone:
            try:
                return datetime.datetime.now(zoneinfo.ZoneInfo(self.timezone))
            except Exception as e:
                self.logger.error(f"Invalid timezone: {self.timezone}, error={e}")
        return datetime.datetime.now()

    def _vault_root(self) -> Path:
        watcher = self._get_component_optional(ComponentEnum.FILE_WATCHER, "default")
        if watcher is None:
            vr = getattr(self.file_store, "vault_root", None)
            return Path(vr).resolve() if vr else Path.cwd().resolve()
        return Path(getattr(watcher, "watch_path", ".")).resolve()

    async def execute(self):
        assert self.context is not None
        mode: str = (self.context.get("mode") or "distill").lower()
        if mode == "log":
            await self._delegate_to_sync()
            return
        if mode != "distill":
            self.context.response.success = False
            _set_answer(self.context, {
                "error": f"unknown mode {mode!r}; expected 'log' or 'distill'",
            })
            return

        content: str = self.context.get("content", "") or ""
        hint: str = self.context.get("hint", "") or ""
        target_path: str = self.context.get("target_path") or ""
        metadata: dict = dict(self.context.get("metadata") or {})
        related_paths: list[str] = list(self.context.get("related_paths") or [])

        assert content, "content is required"

        # Auto-discover wikilink targets in content as a hint for the agent.
        for link in extract_wikilinks(content):
            hit = memory_io.resolve_wikilink(self.file_store, link)["path"]
            if hit and hit not in related_paths:
                related_paths.append(hit)

        as_llm = self._get_component_optional(ComponentEnum.AS_LLM, "default", "model")
        if as_llm is None:
            result = self._degraded(target_path, metadata, content)
            self.context.response.success = result.success
            _set_answer(self.context, result.model_dump())
            return

        vault_root = self._vault_root()
        audit: list[dict] = []
        toolkit = build_memory_toolkit(self.app_context, audit=audit, toolkit=self.toolkit)

        agent = ReActAgent(
            name="reme_ingestor",
            model=self.as_llm,
            sys_prompt=self.prompt_format(
                "system_prompt",
                vault_root=str(vault_root),
                protocol=self._protocol,
            ),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = self.prompt_format(
            "user_message",
            today=self._now().strftime("%Y-%m-%d"),
            vault_root=str(vault_root),
            hint=hint or "(none)",
            target_path=target_path or "(none)",
            metadata=json.dumps(_to_jsonable(metadata), ensure_ascii=False),
            related=json.dumps(related_paths, ensure_ascii=False),
            content=content,
        )

        final_msg: Msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
        )
        summary = final_msg.get_text_content() or ""

        result = IngestResult(used_llm=True)
        for entry in audit:
            (result.applied if entry.get("ok") else result.failed).append(entry)
        if not audit and summary.strip().upper().startswith("SKIP"):
            result.skipped = True

        self.context.response.success = result.success
        payload = result.model_dump()
        payload["agent_summary"] = summary
        _set_answer(self.context, payload)

    def _degraded(self, target_path: str, metadata: dict, content: str) -> IngestResult:
        """Without an LLM, only direct create from explicit target_path is
        supported. Useful for tests and bootstrap scripts."""
        result = IngestResult(used_llm=False)
        if not target_path:
            result.failed.append({
                "op": "create",
                "ok": False,
                "error": "degraded path: target_path is required when no LLM is configured",
            })
            return result
        path = Path(target_path)
        if not path.is_absolute():
            path = self._vault_root() / path
        ok, payload = create_file(
            self.file_store, path,
            metadata=metadata, content=content,
        )
        bucket = result.applied if ok else result.failed
        bucket.append({"op": "create", "ok": ok, "path": str(path), "result": payload})
        return result

    async def _delegate_to_sync(self) -> None:
        """Hot-path event-folder upsert — same code path as the standalone
        `sync` step. Lazy-instantiated so we don't pay the construction
        cost on every distill call."""
        from ..mcp.steps.sync import Sync

        if getattr(self, "_sync_step", None) is None:
            self._sync_step = Sync(app_context=self.app_context)
        await self._sync_step(self.context)

