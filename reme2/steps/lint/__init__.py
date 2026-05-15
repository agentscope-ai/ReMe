"""Lint steps — atomic vault-health checks.

Read-only diagnostics for maintainer / CLI / scheduled-job use;
agents typically don't need them in their per-call working set.

Four atomic checks, each does one thing and returns pure data:

    lint:dangling     — FileLinks pointing to non-existent nodes
    lint:orphans      — nodes with no inlinks AND no outlinks
    lint:collisions   — basenames resolving to >1 path (short-link
                        ambiguity)
    lint:schema       — nodes violating frontmatter schema
                        (missing required fields, invalid status)

Maintainer compositions live in ``memory/maintainer.py``; this
package is the underlying primitives. Each step is also independently
MCP/agent callable for ad-hoc checks. Bind via
``memory.lint_toolkit.build_lint_toolkit``.
"""

from . import dangling  # noqa: F401  -- @R.register("lint:dangling")
from . import orphans  # noqa: F401  -- @R.register("lint:orphans")
from . import collisions  # noqa: F401  -- @R.register("lint:collisions")
from . import schema  # noqa: F401  -- @R.register("lint:schema")
