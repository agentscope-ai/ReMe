"""Parse a complete workspace JSON file without modifying it."""

import json

import aiofiles

from ...base_step import BaseStep
from ...file_io._path import _check_path_permission, resolve_path
from ....components import R
from ....utils.file_utils import get_path_lock


@R.register("check_json_step")
class CheckJsonStep(BaseStep):
    """Return the parsed JSON value or a path/line/column error."""

    async def execute(self):
        raw = self.context.get("path")
        try:
            limit = self.max_file_bytes()
            if limit < 1:
                raise ValueError("max_file_bytes must be positive")
            target, error = resolve_path(self.workspace_path, raw)
            if error:
                raise ValueError(error)
            if not _check_path_permission(self.workspace_path, target, self.context.get("_allowed_paths")):
                raise PermissionError("no permission to read this file")
            async with await get_path_lock(target):
                async with aiofiles.open(target, "rb") as handle:
                    content = await handle.read(limit + 1)
                if len(content) > limit:
                    raise ValueError(f"file exceeds read limit of {limit} bytes")
                self.context.response.answer = json.loads(content)
        except (OSError, ValueError, RecursionError) as exc:
            self.context.response.success = False
            self.context.response.answer = {
                "path": raw,
                "line": exc.lineno if isinstance(exc, json.JSONDecodeError) else None,
                "column": exc.colno if isinstance(exc, json.JSONDecodeError) else None,
                "error": str(exc),
            }
        return self.context.response
