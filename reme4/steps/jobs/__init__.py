"""Jobs steps — composite ReAct-agent-driven workflows.

Two steps:

    digester     — cold-write: distill daily notes into digest/ (R-M-W via a ReAct agent).
    synchronizer  — hot-write: persist in-progress task as a daily note.
"""

from . import digester  # noqa: F401  -- @R.register("digester")
from . import synchronizer  # noqa: F401  -- @R.register("synchronizer")
