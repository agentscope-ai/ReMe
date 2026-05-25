"""Daily-aware steps — CRUD on note md + day-level index.

A daily note is the single file ``daily/<YYYY-MM-DD>/<slug>.md``.
The day-level index ``daily/<YYYY-MM-DD>.md`` aggregates that day's
notes into a richer overview page (note list with name / description).
The index is a derived artifact — its source of truth lives in each
note's frontmatter; refreshes are idempotent and preserve manual
annotations in marker-delimited sections.

Tool boundary. The daily module exposes only the operations whose
shape is note- or day-specific:

* ``daily_read_step``    — read a note by ``slug + date``; returns the
  body in ``answer`` and the parsed frontmatter as a dict in metadata,
  so callers skip a separate ``frontmatter_read`` round-trip.
* ``daily_write_step``   — write the full body + frontmatter for
  ``daily/<date>/<slug>.md`` in one shot. Validates the slug, mkdirs
  the day folder, refreshes the day index. ``mode="create"`` (default)
  is idempotent skip-if-exists; ``mode="overwrite"`` is unconditional.
* ``daily_list_step``    — pure read of the notes under a single day
  (defaults to today); returns ``{date, notes: [{path, slug, name,
  description}, ...]}``. Does **not** touch the day index — call
  ``daily_reindex`` explicitly when the rollup page needs rebuilding.
* ``daily_reindex_step`` — explicit idempotent rebuild of a day's
  index (historical backfill, drift recovery, batch-write reindex).

Body mid-edits / appends / arbitrary-path reads go through the generic
``read`` / ``write`` / ``append`` / ``edit`` steps. Frontmatter slice
mutations go through ``frontmatter_update`` / ``frontmatter_delete``.
The day-index is rebuilt explicitly via ``daily_reindex`` after a
batch of mutations.
"""

# Module name 'list' mirrors its tool name.
# pylint: disable=redefined-builtin

from . import read  # noqa: F401 -- @R.register("daily_read_step")
from . import write  # noqa: F401 -- @R.register("daily_write_step")
from . import list  # noqa: F401 -- @R.register("daily_list_step")
from . import reindex  # noqa: F401 -- @R.register("daily_reindex_step")
