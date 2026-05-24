"""``daily_resolve`` — ensure a daily workspace folder exists; return its vault path.

A daily workspace is the folder ``daily/<YYYY-MM-DD>/<name>/``. This step
creates it if missing and returns the vault-relative path.

Input is a single ``name`` (the workspace name). It must be safe to use as a
folder name on all platforms — Windows is the strictest, so we validate
against its rules:

- no reserved characters: ``< > : " / \\ | ? *`` or control chars (``\\x00-\\x1f``)
- no reserved device names: ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1-9``, ``LPT1-9``
- no trailing ``.`` or whitespace
- no leading/trailing whitespace
- non-empty

Idempotent: if the folder already exists, returns ``{created: False,
message: ...}`` so the caller knows to read-modify rather than overwrite.
"""

from __future__ import annotations

import re
from datetime import date as _date
from pathlib import Path

from ..base_step import BaseStep
from ...utils import set_answer

from ...components import R
from ...enumeration import ComponentEnum


_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _today_iso() -> str:
    return _date.today().isoformat()


def _validate_name(name: str) -> str | None:
    """Return an error string if ``name`` violates Windows folder-name rules; else ``None``."""
    if not name:
        return "name is required"
    if name != name.strip():
        return f"name cannot have leading or trailing whitespace: {name!r}"
    if _INVALID_CHARS.search(name):
        return f'name contains invalid characters (one of < > : " / \\ | ? * or a control char): {name!r}'
    if name.endswith("."):
        return f"name cannot end with '.': {name!r}"
    # Windows reserves these device names with or without an extension (CON.txt also forbidden).
    stem = name.split(".", 1)[0].upper()
    if stem in _RESERVED_NAMES:
        return f"name is a Windows-reserved device name: {name!r}"
    return None


@R.register("daily_resolve_step")
class DailyResolveStep(BaseStep):
    """Ensure ``daily/<today>/<name>/`` exists; return its vault-relative path."""

    component_type = ComponentEnum.STEP

    async def execute(self):
        assert self.context is not None
        name: str = self.context.get("name", "") or ""

        err = _validate_name(name)
        if err:
            payload = {"error": err}
            self.context.response.success = False
            set_answer(self.context, payload)
            return

        day = _today_iso()
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        folder_rel = f"{daily_dir}/{day}/{name}"
        vault_dir = Path(self.file_store.vault_path or ".")
        folder_abs = (vault_dir / folder_rel).resolve()

        already_existed = folder_abs.is_dir()
        if not already_existed:
            folder_abs.mkdir(parents=True, exist_ok=True)

        payload: dict = {
            "date": day,
            "name": name,
            "path": folder_rel,
            "created": not already_existed,
        }
        if already_existed:
            payload["message"] = f"workspace already exists at {folder_rel}"

        self.context.response.success = True
        set_answer(self.context, payload)
