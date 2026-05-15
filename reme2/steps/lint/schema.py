"""``lint:schema`` — find nodes whose frontmatter violates the memory schema.

Atomic vault-health check: required-key presence + ``status`` enum
validity. Mirrors the validator that fires inside ``memory_create``;
keep ``_REQUIRED_META_KEYS`` and ``_VALID_STATUS`` in sync with the
path-template + status-state-machine rules in ``memory_toolkit``.

Returns:

    {count: int, findings: [
        {path, errors: [<error_msg>, ...]},
        ...
    ]}
"""

from __future__ import annotations

from agentscope.tool import ToolResponse

from ..base_step import BaseStep
from ..runtime_response import _set_answer, _tool_response

from ...component import R
from ...enumeration import ComponentEnum


_REQUIRED_META_KEYS = ("title", "lifecycle", "scope", "source", "role")
_VALID_STATUS = {"active", "distilled", "archived"}


def _scan_schema(file_store) -> list[dict]:
    out: list[dict] = []
    for path, node in file_store.file_nodes.items():
        meta = node.front_matter.model_dump()
        errs: list[str] = []
        missing = [k for k in _REQUIRED_META_KEYS if not meta.get(k)]
        if missing:
            errs.append(f"missing required: {missing}")
        status = meta.get("status")
        if status is not None and status not in _VALID_STATUS:
            errs.append(f"invalid status: {status!r} (expected one of {sorted(_VALID_STATUS)})")
        if errs:
            out.append({"path": path, "errors": errs})
    return out


@R.register("lint:schema")
class LintSchema(BaseStep):
    """List nodes whose frontmatter violates the memory schema."""

    component_type = ComponentEnum.STEP

    audit: list[dict] | None = None

    async def execute(self):
        assert self.context is not None
        findings = _scan_schema(self.file_store)
        _set_answer(self.context, {"count": len(findings), "findings": findings})

    async def lint_schema(self) -> ToolResponse:
        """List nodes whose frontmatter violates the memory schema."""
        findings = _scan_schema(self.file_store)
        return _tool_response(
            "lint:schema", True,
            {"count": len(findings), "findings": findings},
            audit=self.audit,
        )
