"""``auto_memory`` — record conversation facts into a daily note.

Calls ``daily_create`` as a system call to provision the note path,
then hands off to a ReAct agent that reads existing content (if any),
decides what to preserve, and writes the note via ``read`` / ``edit``
/ ``frontmatter_update`` / ``write`` tools.

Inputs (from RuntimeContext):
    messages (list[Msg], required): conversation slice to inspect.
    session_id (str, optional): passed to daily_create to determine
        the note path.
    memory_hint (str, optional): caller-supplied hint for the agent.
    timezone (str, optional): IANA timezone for date resolution.

Output (written to context.response):
    answer: one-line summary from the agent.
    metadata: {path, created}.
"""

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import Toolkit

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R


@R.register("auto_memory_step")
class AutoMemoryStep(BaseStep):
    """Record conversation facts into a daily note via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    async def execute(self):
        assert self.context is not None
        messages: list[Msg] = [
            item if isinstance(item, Msg) else Msg.model_validate(item) for item in self.context.get("messages", [])
        ]
        session_id: str = self.context.get("session_id", "")
        memory_hint: str = self.context.get("memory_hint", "")
        current = now(self.context.get("timezone"))

        if not messages:
            self.context.response.success = True
            self.context.response.answer = "Skipped: no messages supplied"
            return

        create_response = await self.run_job("daily_create", session_id=session_id)
        if not create_response.success:
            self.context.response.success = False
            self.context.response.answer = f"daily_create failed: {create_response.answer}"
            return

        note_path: str = create_response.metadata["path"]
        created: bool = create_response.metadata["created"]

        toolkit = Toolkit()
        for job_name in self.agent_tools:
            self.add_as_tool(toolkit, job_name)

        agent = Agent(
            name="auto_memory",
            model=self.llm,
            system_prompt=self.prompt_format("system_prompt"),
            toolkit=toolkit,
        )

        template_key = "user_message_create" if created else "user_message_update"
        user_message: str = self.prompt_format(
            template_key,
            today=current.strftime("%Y-%m-%d"),
            vault_dir=str(self.file_store.vault_path),
            note=memory_hint or "(none)",
            note_path=note_path,
            history=format_history(messages),
        )

        final_msg: Msg = await agent.reply(Msg(name="reme", role="user", content=[TextBlock(text=user_message)]))

        self.context.response.success = True
        self.context.response.answer = (final_msg.get_text_content() or "").strip()
        self.context.response.metadata.update({"path": note_path, "created": created})
