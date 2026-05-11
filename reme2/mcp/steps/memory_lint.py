"""memory_lint — read-only projection of the Maintainer's lint findings.

Tier A surface for the Maintainer service. The host agent calls this
to discover what's wrong with the vault (broken wikilinks, schema
violations, stem collisions) and decides what to do with each finding
using the existing memory_* write primitives.

Equivalent to invoking `Maintainer.execute(ops=["lint"], dry_run=True)`
but with a focused response shape and a tighter parameter surface — the
agent doesn't need to know about decay/merge/split knobs.
"""

from __future__ import annotations

from ...component import R
from ...component.base_step import BaseStep
from ...component.runtime_response import _set_answer
from ...enumeration import ComponentEnum
from ...memory.maintainer import Maintainer


@R.register("memory_lint")
class MemoryLint(BaseStep):
    """Run the Maintainer's lint pass and surface findings only.

    Inputs (RuntimeContext, all optional):
        target_prefix (str): restrict scan to relpaths starting with
            this prefix (e.g. "events/2026-05-09/").

    Output (context.response.answer):
        {
          "scanned": int,                 # files inspected
          "findings": [LintFinding, ...], # each {path, kind, detail}
          "target_prefix": str,
          "ran_at": iso,
        }
    """

    async def execute(self):
        assert self.context is not None
        target_prefix = str(self.context.get("target_prefix") or "")

        # Delegate to a Maintainer step. Force ops=["lint"] + dry_run so
        # we never mutate. The Maintainer reads these from the context
        # and produces a full audit; we narrow the response shape below.
        if getattr(self, "_maintainer", None) is None:
            self._maintainer = R.get(ComponentEnum.STEP, "maintainer")(
                app_context=self.app_context,
            )
        self.context["ops"] = ["lint"]
        self.context["dry_run"] = True
        self.context["target_prefix"] = target_prefix
        await self._maintainer(self.context)

        # The Maintainer wrote a full audit to context.response.answer
        # (proposed/plan/applied/skipped/failed/...). For lint, all the
        # action lives in `proposed` (LintFindings never get applied or
        # dropped). Reshape into a focused response.
        import json
        raw = self.context.response.answer
        audit = json.loads(raw) if isinstance(raw, str) else (raw or {})
        findings = audit.get("proposed") or []

        _set_answer(self.context, {
            "scanned": audit.get("scanned", 0),
            "findings": findings,
            "target_prefix": target_prefix,
            "ran_at": audit.get("ran_at", ""),
        })
        self.context.response.success = True
