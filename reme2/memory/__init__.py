"""Memory subsystem — the three services on top of the core engine.

Per the architecture blueprint:

    - retriever.py   Read service. V + K + Graph BFS fusion + intent
                     routing. Registered as a Step (`backend: hybrid`);
                     MCP step shells in `reme2.mcp.steps.memory_retriever`
                     instantiate it on demand.
    - ingestor.py    Cold-write service. LLM-driven R-M-W curator.
                     Triggered on explicit handoff (task end / SessionEnd).
    - maintainer.py  Treatment service. Background Merge / Split / Decay /
                     Lint, woken by cron or thresholds.
    - summarizer.py  Auxiliary used by the services.

Hot-write MCP step shells (sync, memory_*) live in
`reme2.mcp.steps`, NOT here — they bypass services and write MFS
directly. Importing this package triggers @R.register on the three
services so configs that name them resolve at boot.
"""

from . import retriever   # noqa: F401  -- runs @R.register("hybrid")
from . import maintainer  # noqa: F401  -- runs @R.register("maintainer")
