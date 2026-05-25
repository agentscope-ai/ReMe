"""Daily-aware steps — note + day-level index, on top of generic file ops.

A daily note is the single file ``daily/<YYYY-MM-DD>/<name>.md``.
The day-level index ``daily/<YYYY-MM-DD>.md`` aggregates that day's
notes into a richer overview page: note list with name/description.
The index is a derived artifact — its source of truth lives in each
note's frontmatter and outlinks; refreshes are idempotent and preserve
manual annotations in marker-delimited sections.

Tool boundary. The daily module exposes only the operations whose shape
is note- or day-specific:

* ``daily_resolve_step`` — note path resolver: ensures the day
  folder ``daily/<today>/`` exists and returns the vault-relative
  path ``daily/<today>/<name>.md``. Pure path-shape helper — no
  body, frontmatter, or index writes (those go through the generic
  CRUD + reindex steps).
* ``daily_create_step``  — write the note stub ``daily/<date>/<slug>.md``
  with a minimal ``name`` frontmatter and refresh the day index.
  Idempotent: existing file is left untouched; the index still
  refreshes (cheap self-healing).
* ``daily_list_step``    — list the notes under a single day
  (defaults to today); returns ``{date, notes: [{path, name,
  description}, ...]}``. Also rebuilds ``daily/<date>.md`` as a side
  effect (idempotent — the freshly-rendered inventory is what callers
  want). Read view of the same operation ``daily_reindex_step`` exposes
  from the write side.
* ``daily_reindex_step`` — explicit, idempotent rebuild of a day's index
  (historical backfill, drift recovery, batch-create reindex). Returns
  the write-result fields ``{date, path, created, notes_count}``.

Body reads / writes / appends / overwrites all go through the generic
``file_read`` / ``file_write`` tools. Frontmatter edits go through
``property:update``. The day-index is rebuilt explicitly via
``daily_reindex`` after a batch of mutations.
"""

# Module name 'list' mirrors its tool name.
# pylint: disable=redefined-builtin

from . import resolve  # noqa: F401 -- @R.register("daily_resolve_step")
from . import create  # noqa: F401 -- @R.register("daily_create_step")
from . import list  # noqa: F401 -- @R.register("daily_list_step")
from . import reindex  # noqa: F401 -- @R.register("daily_reindex_step")
