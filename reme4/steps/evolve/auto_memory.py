"""``auto_memory`` — record conversation facts into a daily note.

Calls ``daily_create`` as a system call to provision the note path,
then hands off to a ReAct agent that reads existing content (if any),
decides what to preserve, and writes the note via ``read`` / ``edit``
/ ``frontmatter_update`` / ``write`` tools.

Inputs (from RuntimeContext):
    messages (list[Msg], optional): conversation slice to inspect.
        Mutually exclusive with ``transcript_path``; if both are
        provided, ``messages`` wins.
    transcript_path (str, optional): absolute path to a Claude Code
        transcript JSONL file. When provided (and ``messages`` is
        empty), the step parses the file via
        :func:`reme4.utils.transcript.load_messages_from_transcript`
        and proceeds as if those were the messages. This is what the
        ``reme-service`` plugin's PreCompact / SessionEnd hooks pass
        in directly via ``type: mcp_tool``, replacing the temporary
        spawn-subagent bridge.
    session_id (str, optional): passed to daily_create to determine
        the note path.
    memory_hint (str, optional): caller-supplied hint for the agent.
    timezone (str, optional): IANA timezone for date resolution.

Output (written to context.response):
    answer: one-line summary from the agent.
    metadata: {path, created, n_messages, transcript_path?}.
"""

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import Toolkit

from ._evolve import format_history, now
from ..base_step import BaseStep
from ...components import R
from ...utils.transcript import load_messages_from_transcript


@R.register("auto_memory_step")
class AutoMemoryStep(BaseStep):
    """Record conversation facts into a daily note via an Agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agent_tools: list[str] = ["read", "edit", "frontmatter_update", "write"]

    @staticmethod
    def _to_msg(item) -> Msg:
        if isinstance(item, Msg):
            return item
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            item = {**item, "content": [{"type": "text", "text": item["content"]}]}
        return Msg.model_validate(item)

    async def execute(self):
        assert self.context is not None
        raw_messages = self.context.get("messages") or []
        transcript_path: str = self.context.get("transcript_path", "") or ""
        session_id: str = self.context.get("session_id", "")
        memory_hint: str = self.context.get("memory_hint", "")
        current = now(self.context.get("timezone"))

        # If caller passed transcript_path (the canonical Claude Code hook
        # input) instead of messages, parse it here so the rest of the step
        # stays unchanged.
        if not raw_messages and transcript_path:
            raw_messages = load_messages_from_transcript(transcript_path)
            self.logger.info(
                f"[{self.name}] loaded {len(raw_messages)} messages from transcript_path={transcript_path}",
            )

        messages: list[Msg] = [self._to_msg(item) for item in raw_messages]

        if not messages:
            self.context.response.success = True
            reason = (
                f"Skipped: no messages in transcript_path={transcript_path}"
                if transcript_path
                else "Skipped: no messages supplied"
            )
            self.context.response.answer = reason
            self.context.response.metadata.update(
                {"n_messages": 0, "transcript_path": transcript_path},
            )
            self.logger.info(f"[{self.name}] skipped: {reason} session_id={session_id!r}")
            return

        create_response = await self.run_job("daily_create", session_id=session_id)
        if not create_response.success:
            self.context.response.success = False
            self.context.response.answer = f"daily_create failed: {create_response.answer}"
            self.logger.info(f"[{self.name}] daily_create failed session_id={session_id!r}")
            return

        note_path: str = create_response.metadata["path"]
        created: bool = create_response.metadata["created"]
        self.logger.info(
            f"[{self.name}] note_path={note_path} created={created} "
            f"messages={len(messages)} hint={'yes' if memory_hint else 'no'}",
        )

        toolkit = Toolkit()
        for job_name in self.agent_tools:
            self.add_as_tool(toolkit, job_name)

        agent = Agent(
            name="auto_memory",
            model=self.as_llm,
            system_prompt=self.prompt_format("system_prompt"),
            toolkit=toolkit,
            state=AgentState(
                permission_context=PermissionContext(
                    mode=PermissionMode.BYPASS,
                ),
            ),
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
        self.context.response.metadata.update(
            {
                "path": note_path,
                "created": created,
                "n_messages": len(messages),
                "transcript_path": transcript_path,
            },
        )
        self.logger.info(f"[{self.name}] done note_path={note_path}")
