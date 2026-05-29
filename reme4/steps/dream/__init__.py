"""dream — auto-dream pipeline: classify + integrate via constrained digest tools.

Four steps:

    dreamer            — 2-phase ReAct workflow (extract memory sub-units,
                         then integrate per sub-unit via the digest tools).
    cron_dreamer       — daily wrapper around dreamer; scans today's
                         daily/ + resource/ files and runs dream_one on each.
    digest_write_step  — constrained WriteStep that creates a new
                         digest/<bucket>/<slug>.md.
    digest_edit_step   — constrained EditStep that find-and-replaces in
                         an existing digest node, enforcing E-1 edge
                         conservation.
"""

from . import cron_dreamer  # noqa: F401  -- @R.register("cron_dreamer_step")
from . import digest_edit  # noqa: F401  -- @R.register("digest_edit_step")
from . import digest_write  # noqa: F401  -- @R.register("digest_write_step")
from . import dreamer  # noqa: F401  -- @R.register("dreamer_step")
