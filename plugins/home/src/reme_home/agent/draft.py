"""Bounded reusable lessons owned by one Home workflow invocation."""

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from agentscope.tool import FunctionTool

from reme.utils.qwen_token_tools import QwenTokenTools

from ..evaluation.component import blocking

MAX_DRAFTS = 6
RECOMMENDED_DRAFT_TOKENS = 200
HARD_DRAFT_TOKENS = RECOMMENDED_DRAFT_TOKENS * 110 // 100


@dataclass
class DraftStore:
    """An isolated FIFO notebook; it survives context compression, not restart."""

    entries: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_DRAFTS))
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def add(self, entry: str) -> None:
        """Append one lesson, evicting the oldest when full."""
        with self._lock:
            self.entries.append(entry)

    def read(self) -> list[str]:
        """Return a snapshot ordered from oldest to newest."""
        with self._lock:
            return list(self.entries)


def create_draft_tools(store: DraftStore | None = None, *, token_tools: QwenTokenTools | None = None):
    """Bind notebook tools to a fresh store or the host's diagnostic store."""
    bound_store = store if store is not None else DraftStore()
    tokenizer = token_tools if token_tools is not None else QwenTokenTools()

    async def add_draft(experience: str):
        """Save one self-contained, reusable lesson of at most 200 tokens."""
        if not isinstance(experience, str) or not experience.strip():
            raise ValueError("experience must be a non-empty string")
        saved, _count, _truncated = await blocking(tokenizer.truncate, experience.strip(), HARD_DRAFT_TOKENS)
        bound_store.add(saved)
        return {"status": "saved"}

    def read_draft():
        """Read all reusable lessons, ordered from oldest to newest."""
        entries = bound_store.read()
        return {"entries": entries, "entry_count": len(entries)}

    return [
        FunctionTool(
            add_draft,
            name="add_draft",
            description=(
                "Save one important, reusable lesson discovered through investigation or repeated trial and "
                "error, such as how to inspect data or use a tool effectively. Add only one lesson per call, "
                "at most 200 tokens. At most six lessons are kept; adding another evicts the oldest."
            ),
            is_read_only=True,
            is_concurrency_safe=False,
        ),
        FunctionTool(
            read_draft,
            name="read_draft",
            description="Read all reusable lessons saved by this agent, from oldest to newest.",
            is_read_only=True,
            is_concurrency_safe=False,
        ),
    ], bound_store
