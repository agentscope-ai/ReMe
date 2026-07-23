"""Backend-neutral agent session commands."""

from dataclasses import dataclass

from .base_agent_wrapper import BaseAgentWrapper


@dataclass(frozen=True)
class SessionCommandResult:
    """Result of a handled session command."""

    session_id: str | None
    answer: str


async def handle_session_command(
    wrapper: BaseAgentWrapper,
    text: str,
    session_id: str | None,
) -> SessionCommandResult | None:
    """Handle a supported command, or return ``None`` for ordinary input."""
    if text == "/clear":
        return SessionCommandResult(None, "✅ Conversation cleared. The next message will start a new session.")
    if text != "/compact":
        return None
    if not session_id:
        return SessionCommandResult(None, "No active conversation to compact.")
    await wrapper.compact_session(session_id)
    return SessionCommandResult(session_id, "✅ Conversation compaction requested.")
