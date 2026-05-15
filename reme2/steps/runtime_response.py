"""Helpers for serializing Step output onto `RuntimeContext.response`
and into `agentscope.tool.ToolResponse`.

Lives next to `runtime_context.py` because both are about the BaseStep
interface — the response side, specifically. Used by every Step that
returns a JSON-shaped payload (agent toolkit, lint toolkit, the three
memory services).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _to_jsonable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    return value


def _set_answer(context, payload) -> None:
    context.response.answer = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)


def _tool_response(
    op: str,
    ok: bool,
    payload: Any,
    audit: list[dict] | None = None,
) -> ToolResponse:
    """Wrap a tool-method result as `ToolResponse` and optionally
    append an audit row. Shared by every BaseStep's tool-method
    surface so each toolkit doesn't reinvent serialization."""
    if audit is not None:
        entry = {"op": op, "ok": ok}
        if isinstance(payload, dict):
            entry.update(payload)
        else:
            entry["result"] = payload
        audit.append(entry)
    text = json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2)
    return ToolResponse(content=[TextBlock(type="text", text=text)])
