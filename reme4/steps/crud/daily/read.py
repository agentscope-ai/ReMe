"""``daily_read`` — read a daily note by slug + date; return body + parsed frontmatter + link context.

Convenience wrapper around the generic ``read`` step for the
``daily/<YYYY-MM-DD>/<slug>.md`` path shape. The value over a raw
``file_read`` is three-fold:

* slug validation (Windows-safe filename rules) up front
* frontmatter parsed into a dict in metadata, so callers don't need a
  separate ``frontmatter_read`` round-trip
* outlink / inlink graph context expanded by default — the answer
  body is followed by a ``→`` / ``←`` block so agents see neighbors
  without a separate search round-trip

Inputs:
    slug (required, validated)
    date (default today, ISO ``YYYY-MM-DD``)
    expand_links (default True) — set False to skip link expansion
    max_links_per_direction (default 10) — neighbor cap per direction

Outputs:
    answer = note body (frontmatter stripped) + appended link block when non-empty
    metadata = {date, slug, path, exists, frontmatter: dict, link_expansion: dict}

For arbitrary-path reads or ranged reads use the generic ``read`` step.
"""

from datetime import date as _date

import frontmatter

from ._daily_io import validate_slug
from .._file_io import read_file_safe
from ...base_step import BaseStep
from ....components import R
from ....utils.link_expansion import expand_links, render_expansion_lines


@R.register("daily_read_step")
class DailyReadStep(BaseStep):
    """Read ``daily/<date>/<slug>.md`` → body + parsed frontmatter."""

    def _fail(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):
        assert self.context is not None
        slug: str = self.context.get("slug", "") or ""
        day: str = (self.context.get("date") or "").strip() or _date.today().isoformat()

        err = validate_slug(slug)
        if err:
            self._fail(err)
            return None

        daily_dir = self.app_context.app_config.daily_dir if self.app_context is not None else "daily"
        path_rel = f"{daily_dir}/{day}/{slug}.md"
        path_abs = (self.vault_path / path_rel).resolve()

        if not path_abs.is_file():
            self._fail(
                f"note {path_rel} does not exist",
                date=day,
                slug=slug,
                path=path_rel,
                exists=False,
            )
            return None

        try:
            text = await read_file_safe(path_abs)
        except Exception as e:  # pylint: disable=broad-except
            self._fail(f"read failed: {e}", date=day, slug=slug, path=path_rel)
            return None

        post = frontmatter.loads(text)
        body = post.content
        meta = dict(post.metadata or {})

        link_expansion_data: dict = {}
        if self.context.get("expand_links", True):
            max_per_dir = int(self.context.get("max_links_per_direction", 10))
            link_expansion_data = await expand_links(self.file_store, [path_rel], max_per_dir)

        expansion_lines = render_expansion_lines(link_expansion_data.get(path_rel, {}))
        if expansion_lines:
            body = body.rstrip() + "\n\n" + "\n".join(expansion_lines)

        self.context.response.success = True
        self.context.response.answer = body
        self.context.response.metadata.update(
            {
                "date": day,
                "slug": slug,
                "path": path_rel,
                "exists": True,
                "frontmatter": meta,
                "link_expansion": link_expansion_data,
            },
        )
        self.logger.info(
            f"[{self.name}] read path={path_rel} "
            f"bytes={len(body.encode('utf-8'))} fm_keys={list(meta)} "
            f"links={'on' if link_expansion_data else 'off'}",
        )
        return self.context.response
