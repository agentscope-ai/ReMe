"""Pair and rebuild AgentScope blocks without rewriting conversation text."""

from collections import defaultdict
from dataclasses import dataclass
import json

from ...schema.compaction import CompactionRequest


@dataclass(frozen=True)
class ToolPair:
    """Block locations remain valid until all decisions have been collected."""

    id: str
    call_id: str
    tool: str
    input: dict
    call_location: tuple[int, int]
    result_location: tuple[int, int]
    result_chars: int
    protected: bool
    source_ref: str | None


def result_text(block: dict) -> str | None:
    """Only entirely textual outputs can be shortened."""
    output = block.get("output")
    if isinstance(output, str):
        return output
    if (
        isinstance(output, list)
        and output
        and all(
            isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str)
            for item in output
        )
    ):
        return "\n".join(item["text"] for item in output)
    return None


def collect_pairs(request: CompactionRequest) -> list[ToolPair]:
    """Ambiguous, incomplete, failed, unknown, or protected calls are never candidates."""
    calls, results = defaultdict(list), defaultdict(list)
    messages = request.messages
    recent = len(messages) - request.options.preserve_recent_messages
    protected = {
        index
        for index, message in enumerate(messages)
        if index == 0
        or index >= recent
        or message.get("role") == "system"
        or message.get("id") in request.protected_message_ids
    }
    for mi, message in enumerate(messages):
        for bi, block in enumerate(message["content"]):
            block_id = block.get("id")
            if not isinstance(block_id, str) or not block_id:
                continue
            if block.get("type") == "tool_call":
                calls[block_id].append((mi, bi))
            elif block.get("type") == "tool_result":
                results[block_id].append((mi, bi))
    pairs = []
    for call_id, locations in calls.items():
        if len(locations) != 1 or len(results[call_id]) != 1:
            continue
        ci, ri = locations[0], results[call_id][0]
        call, result = messages[ci[0]]["content"][ci[1]], messages[ri[0]]["content"][ri[1]]
        if ri <= ci:
            continue
        try:
            inputs = json.loads(call.get("input", ""))
            # Python accepts NaN/Infinity and overflowing numbers in JSON text.
            # Such inputs must stay untouched rather than break other candidates.
            json.dumps(inputs, allow_nan=False)
        except (TypeError, ValueError):
            continue
        if not isinstance(inputs, dict):
            continue
        text = result_text(result)
        pinned = (
            ci[0] in protected
            or ri[0] in protected
            or call_id in request.protected_call_ids
            or call.get("name") not in request.eligible_tool_names
            or call.get("state") != "finished"
            or result.get("state") != "success"
            or result.get("name") != call.get("name")
            or text is None
            or call_id not in request.source_refs
        )
        pairs.append(
            ToolPair(
                f"t{len(pairs) + 1}",
                call_id,
                call.get("name", ""),
                inputs,
                ci,
                ri,
                len(text or ""),
                pinned,
                request.source_refs.get(call_id),
            ),
        )
    return pairs


def excerpt(text: str, limit: int) -> str:
    """Keep both ends verbatim; never exceed the character limit."""
    if len(text) <= limit:
        return text
    marker = "\n…\n"
    if limit < len(marker):
        return text[:limit]
    available = limit - len(marker)
    head = (available + 1) // 2
    tail = available - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def preview_result(block: dict, source_ref: str | None, limit: int) -> str | None:
    """Build one recoverable replacement without copying or changing the source history."""
    text = result_text(block)
    if text is None or not source_ref:
        return None
    replacement = excerpt(text, limit) + f"\n[Compacted output. Recover from: {source_ref}]"
    return replacement if len(replacement) < len(text) else None
