"""Helpers for serializing Step output onto `RuntimeContext.response`.

Lives next to `runtime_context.py` because both are about the BaseStep
interface — the response side, specifically. Used by every Step that
returns a JSON-shaped payload (memory_*, sync, the three memory
services).
"""

from __future__ import annotations

import json
from datetime import date, datetime


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
