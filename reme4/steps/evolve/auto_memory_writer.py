"""``auto_memory_writer`` — execute daily-note upserts.

Reads the ``memory_updates`` list produced by ``auto_memory_planner``
from ``context.response.metadata['memory_updates']``, then iterates
over each ``{path, description}`` task: decides UPDATE vs CREATE by
probing the vault, and writes the note via ``frontmatter_read`` /
``frontmatter_update`` / ``read`` / ``edit`` / ``write``.

A fresh ReAct agent is created per task to keep conversations isolated.

Inputs (from RuntimeContext):
    messages (list[Msg], required): conversation slice (context).
    memory_hint (str, optional): caller-supplied note hint.
    response.metadata['memory_updates'] (list[dict]): planner output.

Output (written to context.response):
    answer: one line per task — ``<action> <path>``.
    metadata['written_count']: number of tasks executed.
"""

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R


@R.register("auto_memory_writer_step")
class AutoMemoryWriterStep(BaseStep):
    """Execute note upserts from the planner's task list."""

    def __init__(self, console_enabled: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.console_enabled = console_enabled
        self.writer_tools: list[str] = ["frontmatter_read", "frontmatter_update", "read", "edit", "write"]

    async def execute(self):
        assert self.context is not None
        memory_updates: list[dict] = self.context.response.metadata.get("memory_updates") or []
        if not memory_updates:
            self.context.response.success = True
            self.context.response.answer = "[SKIP] No memory updates to write"
            return

        current = now(self.context.get("timezone"))
        messages: list[Msg] = [
            item if isinstance(item, Msg) else Msg.from_dict(item) for item in self.context.get("messages", [])
        ]
        memory_hint: str = self.context.get("memory_hint", "")

        toolkit = Toolkit()
        for job_name in self.writer_tools:
            self.add_as_tool(toolkit, job_name)

        results: list[str] = []
        for task in memory_updates:
            note_path = task.get("path", "")
            description = task.get("description", "")
            if not note_path or not description:
                continue

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
            result_line = (final_msg.get_text_content() or "").strip()
            results.append(result_line)

        self.context.response.success = True
        self.context.response.answer = "\n".join(results) if results else "[SKIP] No valid tasks"
        self.context.response.metadata.update({"written_count": len(results)})
