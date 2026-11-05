"""Read pending resource packages and retain interesting findings."""

import frontmatter

from ._interest import _InterestStep
from ...components import R


@R.register("find_interest_step")
class FindInterestStep(_InterestStep):
    """The agent writes an optional note and acknowledges only packages it processed."""

    async def execute(self):
        memory = await self._memory()
        before = await self._notes()
        now = self._now()
        stem = f"{self.config_value('daily_dir')}/{now:%Y-%m-%d}/interest_{now:%H%M%S}"
        path, suffix = f"{stem}.md", 1
        while self._path(path).exists():
            path, suffix = f"{stem}_{suffix}.md", suffix + 1
        await self._agent(
            memory,
            {"output_path": path, "existing_interests": self._summaries(before)},
            ["resource_changes", "resource_processed"],
        )
        notes = await self._notes(before)
        if notes.keys() - before.keys() - {path} or any(
            name not in notes or frontmatter.dumps(notes[name]) != frontmatter.dumps(post)
            for name, post in before.items()
        ):
            raise ValueError("Find agent must preserve existing interests and write only output_path")
        self.context.response.answer = {"created": [path for path in notes if path not in before]}
        return self.context.response
