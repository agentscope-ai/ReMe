"""Daily-aware steps — workspace + day-level index, on top of generic file ops.

A daily workspace is the folder ``daily/<YYYY-MM-DD>/<name>/`` plus any
files inside it (a ``<name>.md`` summary note by convention, plus sibling
materials). The day-level index ``daily/<YYYY-MM-DD>.md`` aggregates that
day's workspaces into a richer overview page: workspace list with
name/description. The index is a derived artifact — its source of truth
lives in each workspace's frontmatter and outlinks; refreshes are
idempotent and preserve manual annotations in marker-delimited sections.

Tool boundary. The daily module exposes only the operations whose shape
is workspace- or day-specific:

* ``daily_resolve_step`` — workspace path resolver: ensures the folder
  ``daily/<today>/<name>/`` exists and returns its vault-relative path.
  Pure path-shape helper — no body, frontmatter, or index writes
  (those go through the generic CRUD + reindex steps).
* ``daily_list_step``    — list the workspaces under a single day
  (defaults to today); returns ``{date, workspaces: [{path, name,
  description}, ...]}``. Also rebuilds ``daily/<date>.md`` as a side
  effect (idempotent — the freshly-rendered inventory is what callers
  want). Read view of the same operation ``daily_reindex_step`` exposes
  from the write side.
* ``daily_reindex_step`` — explicit, idempotent rebuild of a day's index
  (historical backfill, drift recovery, batch-create reindex). Returns
  the write-result fields ``{date, path, created, workspaces_count}``.

Body reads / writes / appends / overwrites all go through the generic
``file_read`` / ``file_write`` tools. Frontmatter edits go through
``property:update``. The day-index is rebuilt explicitly via
``daily_reindex`` after a batch of mutations.
"""

# Module name 'list' mirrors its tool name.
# pylint: disable=redefined-builtin

from . import resolve  # noqa: F401 -- @R.register("daily_resolve_step")
from . import list  # noqa: F401 -- @R.register("daily_list_step")
from . import reindex  # noqa: F401 -- @R.register("daily_reindex_step")
