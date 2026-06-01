"""dream — auto-dream pipeline: extract abstractions, then integrate per
sub-unit using bucket-specific Phase 2 prompts.

Two steps:

    dreamer       — 2-phase ReAct workflow (extract memory sub-units
                    tagged with bucket, then integrate per sub-unit
                    via the canonical write/edit tools).
    cron_dreamer  — daily wrapper around dreamer; scans today's
                    daily/ + resource/ files and runs dream_one on each.

Phase 2 uses the canonical ``write`` / ``edit`` jobs directly — no
constrained variants. Bucket placement is prompt-level discipline.
"""

from . import cron_dreamer  # noqa: F401  -- @R.register("cron_dreamer_step")
from . import dreamer  # noqa: F401  -- @R.register("dreamer_step")
