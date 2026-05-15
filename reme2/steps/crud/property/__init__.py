"""Property steps — frontmatter-only CRUD on **markdown** files.

Three Steps:

    property:read    — return the frontmatter dict
    property:update  — merge a patch into the frontmatter
    property:delete  — drop the listed keys

Operates only on ``.md`` files; non-markdown targets get
``error="not markdown"`` and the call is **not** executed.

Body content stays untouched; use the ``crud`` siblings (``read``,
``edit``, ``append``, ``prepend``) for body-level operations. Each
Step here is a pure disk read-modify-write — the watcher / parser
notices the change and refreshes the projections asynchronously.
"""

from . import read  # noqa: F401  -- @R.register("property:read")
from . import update  # noqa: F401  -- @R.register("property:update")
from . import delete  # noqa: F401  -- @R.register("property:delete")
