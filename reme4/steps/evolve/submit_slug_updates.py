"""``submit_slug_updates`` — terminal "submit" tool for the planner agent.

The planner agent (in ``auto_memory``) walks today's daily-note metadata
via ``daily_list``, reads any candidates whose ``description`` looks
relevant via ``read``, and finally hands off its plan by calling this
step exactly once. The payload is a list of ``{slug, description}``
entries:

* **slug** — kebab-case identifier of the daily note to create or
  update. Same slug across calls = same file = upsert.
* **description** — instructions for the writer agent: what to write
  into the body and frontmatter (must spell out the
  Personal / Procedural / Knowledge information to preserve).

The step is intentionally side-effect free: it just validates the
payload, surfaces a one-line confirmation as ``answer`` (so the
planner agent sees that its submission was accepted), and stores
the structured list under ``response.metadata['slug_updates']`` for
the orchestrating caller (the outer auto-memory loop) to iterate.

Calling this step is the planner's termination signal — the caller
treats its successful return as "planning is done, hand the tasks to
the writer agent now". No new note is written here.
"""

from ..base_step import BaseStep
from ...components import R


@R.register("submit_slug_updates_step")
class SubmitSlugUpdatesStep(BaseStep):
    """Validate and surface the planner agent's slug-update task list."""

    async def execute(self):
        assert self.context is not None
        raw = self.context.get("slug_updates") or []
        assert isinstance(raw, list), "slug_updates must be a list"

        cleaned: list[dict] = []
        errors: list[str] = []
        for idx, item in enumerate(raw):
            if not isinstance(item, dict):
                errors.append(f"item[{idx}]: not an object")
                continue
            slug = str(item.get("slug") or "").strip()
            description = str(item.get("description") or "").strip()
            if not slug:
                errors.append(f"item[{idx}]: missing 'slug'")
                continue
            if not description:
                errors.append(f"item[{idx}]: missing 'description'")
                continue
            cleaned.append({"slug": slug, "description": description})

        if errors:
            self.context.response.success = False
            self.context.response.answer = "Error: " + "; ".join(errors)
            self.context.response.metadata.update({"slug_updates": [], "errors": errors})
            return

        self.context.response.success = True
        self.context.response.metadata.update({"slug_updates": cleaned, "count": len(cleaned)})
        if not cleaned:
            self.context.response.answer = "Submitted 0 slug updates (nothing to do)"
            return
        lines = [f"Submitted {len(cleaned)} slug update(s):"]
        lines += [f"- {t['slug']}: {t['description']}" for t in cleaned]
        self.context.response.answer = "\n".join(lines)
