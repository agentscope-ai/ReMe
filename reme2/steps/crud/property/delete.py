"""``property:delete`` — remove keys from a memory file's frontmatter.

Returns both ``deleted`` (keys that were present and removed) and
``missing`` (keys that weren't there) so the agent can tell whether
a no-op happened. The file is rewritten only when at least one key
is actually removed — calling delete with all-missing keys is a
zero-side-effect read.

``path`` accepts a short link or vault-relative path. Short-link
ambiguity is reported via ``error="ambiguous"`` with the candidate
list — the delete is **not** executed in that case.
"""

from __future__ import annotations

import frontmatter
from agentscope.tool import ToolResponse

from ...base_step import BaseStep
from ...runtime_response import _set_answer, _tool_response

from ....component import R
from ....enumeration import ComponentEnum
from ....utils import path_resolver


async def _delete(file_store, path: str, keys: list[str]) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        return {"path": path, "error": "not found"}
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if not keys:
        return {"path": path, "error": "keys is empty"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    deleted: list[str] = []
    missing: list[str] = []
    for k in keys:
        if k in post.metadata:
            del post.metadata[k]
            deleted.append(k)
        else:
            missing.append(k)
    if deleted:
        target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {
        "path": path,
        "deleted": deleted,
        "missing": missing,
        "frontmatter": dict(post.metadata),
    }


@R.register("property:delete")
class PropertyDelete(BaseStep):
    """Remove keys from a memory file's frontmatter."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        keys = self.context.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        payload = await _delete(self.file_store, path, list(keys))
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def property_delete(
        self,
        path: str,
        keys: list[str] | str,
    ) -> ToolResponse:
        """Remove the listed ``keys`` from the frontmatter at ``path``."""
        if isinstance(keys, str):
            keys = [keys]
        payload = await _delete(self.file_store, path, list(keys))
        ok = "error" not in payload
        return _tool_response("property:delete", ok, payload, audit=self.audit)
