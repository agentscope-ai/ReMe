"""``daily_resolve`` — resolve a daily note path; ensure the parent day folder exists.

A daily note is a single markdown file ``daily/<YYYY-MM-DD>/<name>.md``.
This step validates ``name``, makes sure the day folder ``daily/<YYYY-MM-DD>/``
exists (so a subsequent ``file_write`` succeeds), and returns the
vault-relative path to the note file.

Input is a single ``name`` (the note slug). It must be safe to use as a
filename on all platforms — Windows is the strictest, so we validate
against its rules:

- no reserved characters: ``< > : " / \\ | ? *`` or control chars (``\\x00-\\x1f``)
- no reserved device names: ``CON``, ``PRN``, ``AUX``, ``NUL``, ``COM1-9``, ``LPT1-9``
- no trailing ``.`` or whitespace
- no leading/trailing whitespace
- non-empty

Idempotent: returns ``{exists: True}`` when the note file already
exists (caller should read-modify rather than overwrite); otherwise
``{exists: False}`` — the file itself is **not** created here, use
``daily_create`` or ``file_write`` for that.
"""

import re
from datetime import date as _date
from pathlib import Path

from ..base_step import BaseStep

from ...components import R


_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@R.register("daily_resolve_step")
class DailyResolveStep(BaseStep):
    """Ensure ``daily/<today>/`` exists; return the vault-relative path to ``<name>.md``."""

    async def execute(self):
        assert self.context is not None
        name: str = self.context.get("name", "") or ""

        err: str | None = None
        if not name:
            err = "name is required"
        elif name != name.strip():
            err = f"name cannot have leading or trailing whitespace: {name!r}"
        elif _INVALID_CHARS.search(name):
            err = f'name contains invalid characters (one of < > : " / \\ | ? * or a control char): {name!r}'
        elif name.endswith("."):
            err = f"name cannot end with '.': {name!r}"
        # Windows reserves these device names with or without an extension (CON.txt also forbidden).
        elif name.split(".", 1)[0].upper() in _RESERVED_NAMES:
            err = f"name is a Windows-reserved device name: {name!r}"

        if err:
            self.context.response.success = False
            self.context.response.answer = f"Error: {err}"
            self.context.response.metadata.update({"error": err})
            return

        day = _date.today().isoformat()
        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        path_rel = f"{daily_dir}/{day}/{name}.md"
        vault_dir = Path(self.file_store.vault_path or ".")
        path_abs = (vault_dir / path_rel).resolve()
        path_abs.parent.mkdir(parents=True, exist_ok=True)

        exists = path_abs.is_file()
        payload: dict = {
            "date": day,
            "name": name,
            "path": path_rel,
            "exists": exists,
        }
        if exists:
            payload["message"] = f"note already exists at {path_rel}"

        self.context.response.success = True
        verb = "Resolved" if not exists else "Resolved existing"
        self.context.response.answer = f"{verb} note {path_rel}"
        self.context.response.metadata.update(payload)
