"""Frontmatter steps — RUD on the YAML frontmatter slice of a markdown file.

Three Steps:

    frontmatter_read_step    — return the frontmatter dict
    frontmatter_update_step  — merge a patch into the frontmatter
    frontmatter_delete_step  — drop the listed keys

Operates only on ``.md`` files; non-markdown targets get
``error="not markdown"`` and the call is **not** executed.

Body content stays untouched; the sibling ``file`` package owns every
other file-level surface — opaque-byte ops (list / stat / move /
delete / upload / download) and whole-file text ops (read / write /
append / edit). For mid-body edits, use ``file.edit`` (exact string
replacement) or do a read + write round-trip.

Each Step here is a pure disk read-modify-write — the watcher / parser
notices the change and refreshes the projections asynchronously.
"""

from . import read  # noqa: F401  -- @R.register("frontmatter_read_step")
from . import update  # noqa: F401  -- @R.register("frontmatter_update_step")
from . import delete  # noqa: F401  -- @R.register("frontmatter_delete_step")
