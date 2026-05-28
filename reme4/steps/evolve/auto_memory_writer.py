"""``auto_memory_writer`` — execute a single daily-note upsert.

Takes one ``(path, description)`` task from the planner plus the
recent conversation, decides UPDATE vs CREATE by probing the vault,
and writes the note via ``frontmatter_read`` / ``frontmatter_update``
/ ``read`` / ``edit`` / ``write``.

The writer agent is told that the body must preserve every
Personal / Procedural / Knowledge fact named in the planner's
description, and that the frontmatter must be rich (name,
description, tags, type, topics, related, status, created, updated,
etc.).

Inputs (from RuntimeContext):
    messages (list[Msg], required): conversation slice (context).
    memory_hint (str, optional): caller-supplied note hint.
    path (str, required): the daily-note vault-relative path to upsert.
    description (str, required): planner-produced instructions.

Output (written to context.response.answer):
    One line: ``<action> <path>`` where ``<action>`` ∈ {{created, updated}}.
"""

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R


@R.register("auto_memory_writer_step")
class AutoMemoryWriterStep(BaseStep):
    """Execute one note upsert via a ReAct agent."""

    def __init__(self, console_enabled: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.console_enabled = console_enabled
        self.writer_tools: list[str] = ["frontmatter_read", "frontmatter_update", "read", "edit", "write"]

    async def execute(self):
        assert self.context is not None
        note_path = self.context.get("path", "")
        description = self.context.get("description", "")
        current = now(self.context.get("timezone"))
        assert note_path, "path is required"
        assert description, "description is required"

        messages: list[Msg] = [
            item if isinstance(item, Msg) else Msg.from_dict(item)
            for item in self.context.get("messages", [])
        ]
        memory_hint: str = self.context.get("memory_hint", "")

        toolkit = Toolkit()
        for job_name in self.writer_tools:
            self.add_as_tool(toolkit, job_name)

        agent = ReActAgent(
            name="auto_memory_writer",
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
            note_path=note_path,
            description=description,
            history=format_history(messages),
        )

        final_msg: Msg = await agent.reply(Msg(name="reme", role="user", content=user_message))
        self.context.response.success = True
        self.context.response.answer = (final_msg.get_text_content() or "").strip()
        self.context.response.metadata.update({"path": note_path})
