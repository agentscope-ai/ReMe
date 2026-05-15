"""``property:update`` — set frontmatter keys on a markdown file.

Read-modify-write the YAML frontmatter; body content is untouched.
The watcher / parser pick up the change asynchronously.

Free-form kwargs form: ``property:update path=foo.md x=y z=w`` sets
frontmatter ``x`` to ``y`` and ``z`` to ``w``. Each keyword arg
becomes one frontmatter entry.

``path`` accepts a short link or vault-relative path. Short-link
ambiguity is reported via ``error="ambiguous"`` with the candidate
list — the update is **not** executed in that case. Non-markdown
targets return ``error="not markdown"``.
"""

from __future__ import annotations

import frontmatter
from agentscope.tool import ToolResponse

from ...base_step import BaseStep
from ...runtime_response import _set_answer, _tool_response

from ....component import R
from ....enumeration import ComponentEnum
from ....utils import path_resolver


# Context keys that belong to plumbing (request envelope, response
# slot, the path itself) and must never be promoted into a frontmatter
# update.
_RESERVED_CONTEXT_KEYS = {"path", "response", "request", "data"}


async def _update(file_store, path: str, fields: dict) -> dict:
    try:
        target = await path_resolver.resolve_to_absolute(file_store, path)
    except path_resolver.PathAmbiguous as e:
        return {"path": path, "error": "ambiguous", "candidates": e.candidates}
    except path_resolver.PathNotFound:
        return {"path": path, "error": "not found"}
    if not target.is_file():
        return {"path": path, "error": "not found"}
    if target.suffix != ".md":
        return {"path": path, "error": "not markdown"}
    if not fields:
        return {"path": path, "error": "no fields to update"}
    raw = target.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    post.metadata.update(fields)
    target.write_text(frontmatter.dumps(post), encoding="utf-8")
    return {"path": path, "updated": fields}


@R.register("property:update")
class PropertyUpdate(BaseStep):
    """Set frontmatter keys on a markdown file via free-form kwargs."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        path: str = self.context.get("path") or ""
        assert path, "path is required"
        fields = {
            k: v for k, v in self.context.data.items()
            if k not in _RESERVED_CONTEXT_KEYS
        }
        payload = await _update(self.file_store, path, fields)
        self.context.response.success = "error" not in payload
        _set_answer(self.context, payload)

    async def property_update(self, path: str, **kwargs) -> ToolResponse:
        """Set frontmatter entries: each ``key=value`` kwarg becomes one
        frontmatter field on the markdown file at ``path``."""
        payload = await _update(self.file_store, path, dict(kwargs))
        ok = "error" not in payload
        return _tool_response("property:update", ok, payload, audit=self.audit)
