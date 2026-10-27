"""MemLens memory prompts with optional, exact source-line references."""

from agentscope.message import Msg

from reme.steps.evolve.auto_memory import AutoMemoryStep


class MemLensAutoMemoryStep(AutoMemoryStep):
    """Keep core persistence and multimodal input, adding benchmark instructions only."""

    def _format_history(self, messages: list[Msg]) -> str:
        if not self.context.get("include_source_sessions", True):
            return "Source-session access is disabled; record self-contained facts.\n\n" + super()._format_history(
                messages,
            )

        # Core execute has saved the session already. Read physical line numbers
        # rather than assuming input order equals timestamp-sorted JSONL order.
        session_id = self.context.get("session_id", "")
        source = self._session_source_path(session_id)
        # File iteration keeps Unicode line separators inside JSON strings intact.
        with self._session_path(session_id).open(encoding="utf-8") as stream:
            lines = {Msg.model_validate_json(line).id: number for number, line in enumerate(stream, 1) if line.strip()}
        turns = [
            f"[L{lines[msg.id]} | {msg.name or msg.role} @ {msg.created_at}]\n{msg.get_text_content() or ''}"
            for msg in messages
        ]
        return (
            f"Source file: {source}\n"
            f"Cite useful evidence with the full source path, for example [[{source}#L1-L2]], "
            "replacing the line range with the actual physical lines below. Never use bare links such as [[L5]]. "
            "Preserve original Image IDs alongside their facts, and retain the supplied Image source paths.\n\n"
            + "\n\n".join(turns)
        )
