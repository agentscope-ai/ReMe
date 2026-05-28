"""``auto_memory_planner`` — planner for the auto-memory system.

Inspects the recent conversation, surveys today's existing daily notes
via ``daily_list``, reads any candidate that already covers the topic
via ``read``, and emits a list of daily-note upsert tasks
``(path, description)`` through structured output.  The result is
exposed under ``response.metadata['memory_updates']`` for the
orchestrator (``auto_memory``) to feed into ``auto_memory_writer``
one task at a time.

The planner never writes notes itself — planning only.

Inputs (from RuntimeContext):
    messages (list[Msg], required): conversation slice to inspect.
    memory_hint (str, optional): caller-supplied hint to bias filename
        stem selection or disambiguate same-day tasks.

Output (written to context.response):
    answer: one-line human summary of what was planned.
    metadata['memory_updates']: list of ``{path, description}`` dicts.
"""

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from pydantic import BaseModel, Field

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R


class MemoryUpdateTask(BaseModel):
    """One daily-note upsert task emitted by the planner."""

    path: str = Field(description="Vault-relative note path, form `daily/<YYYY-MM-DD>/<kebab-case-stem>.md`. Reuse an existing path to upsert.")
    description: str = Field(description="Flat fact checklist — what to preserve, not how to categorize or format.")


class MemoryUpdatesPlan(BaseModel):
    """Structured output emitted by the planner's finish-tool."""

    memory_updates: list[MemoryUpdateTask] = Field(
        default_factory=list,
        description="List of daily-note upsert tasks; empty means nothing worth persisting.")


@R.register("auto_memory_planner_step")
class AutoMemoryPlannerStep(BaseStep):
    """Plan daily-note upsert tasks via a ReAct agent with structured output."""

    def __init__(self, console_enabled: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.console_enabled = console_enabled
        self.planner_tools: list[str] = ["daily_list", "read"]

    async def execute(self):
        assert self.context is not None
        messages: list[Msg] = [item if isinstance(item, Msg) else Msg.from_dict(item)
                               for item in self.context.get("messages", [])]
        memory_hint: str = self.context.get("memory_hint", "")
        current = now(self.context.get("timezone"))

        if not messages:
            self.context.response.success = True
            self.context.response.answer = "Skipped: no messages supplied"
            self.context.response.metadata.update({"memory_updates": []})
            return

        toolkit = Toolkit()
        for job_name in self.planner_tools:
            self.add_as_tool(toolkit, job_name)

        agent = ReActAgent(
            name="auto_memory_planner",
            model=self.as_llm,
            sys_prompt=self.prompt_format("system_prompt"),
            formatter=self.as_llm_formatter,
            toolkit=toolkit,
        )
        agent.set_console_output_enabled(self.console_enabled)

        user_message: str = self.prompt_format(
            "user_message",
            today=current.strftime("%Y-%m-%d"),
            vault_dir=str(self.file_store.vault_path),
            note=memory_hint or "(none)",
            history=format_history(messages),
        )

        final_msg: Msg = await agent.reply(
            Msg(name="reme", role="user", content=user_message),
            structured_model=MemoryUpdatesPlan,
        )

        meta: dict = final_msg.metadata if isinstance(final_msg.metadata, dict) else {}
        raw_tasks = meta.get("memory_updates") or []
        cleaned: list[dict] = []
        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            description = str(item.get("description") or "").strip()
            if path and description and path.endswith(".md") and ".." not in path.split("/"):
                cleaned.append({"path": path, "description": description})

        self.context.response.success = True
        self.context.response.metadata.update({"memory_updates": cleaned, "count": len(cleaned)})
        if not cleaned:
            self.context.response.answer = "[SKIP] No memory updates planned"
            return

        lines = [f"Planned {len(cleaned)} memory update(s):"]
        lines += [f"- {t['path']}: {t['description']}" for t in cleaned]
        self.context.response.answer = "\n".join(lines)
