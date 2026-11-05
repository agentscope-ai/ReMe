"""Shared helpers for evolve steps."""

import datetime
import zoneinfo

from agentscope.message import Msg

from ...schema import AgentReply, Response


def now(timezone: str | None = None) -> datetime.datetime:
    """Return current datetime in the given IANA timezone, falling back to local."""
    if not timezone:
        return datetime.datetime.now()
    try:
        return datetime.datetime.now(zoneinfo.ZoneInfo(timezone))
    except Exception:
        return datetime.datetime.now()


def format_history(messages: list[Msg], include_timestamp: bool = True) -> str:
    """Render a conversation slice as a human-readable transcript."""
    lines: list[str] = []
    for msg in messages:
        text = (msg.get_text_content() or "").strip()
        if not text:
            continue
        speaker = msg.name or msg.role or "?"
        header = f"[{speaker} @ {msg.created_at}]" if include_timestamp else f"[{speaker}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines) or "(empty)"


def agent_reply_result_text(reply_result: AgentReply) -> str:
    """Require a completed invocation and return its backend-neutral final text."""
    status = reply_result["status"]
    if status != "completed" or reply_result.get("error") is not None:
        raise RuntimeError(reply_result.get("error") or f"Agent reply ended: {status}")
    return reply_result["result"].strip()


def passthrough_response(step, skip_key: str) -> Response:
    """Return a success response when a short-circuit flag is set (INV-7).

    Short-circuited rounds never write interests.yaml or checkpoint catalogs;
    the job still reports success so the skipped round is not counted as a failure.
    """
    assert step.context is not None
    response = step.context.response
    response.success = True
    flag = step.context.get(skip_key)
    if isinstance(flag, dict):
        reason = str(flag.get("reason") or "skipped")
    else:
        reason = str(flag or "skipped")
    response.answer = f"Skipped: {reason}"
    step.logger.info(f"[{step.name}] short-circuit via {skip_key!r} reason={reason}")
    return response
