"""Forward ``python -m validation`` to ``validation.run.main``.

This module contains no scheduling or validation logic. The CLI adapter is in
``validation/run.py`` and the implementation is in ``validation/evaluator.py``.
The separate top-level ``meta-reme/run.py`` prepares a complete workspace
before invoking that same validation implementation.
"""

from .run import main

if __name__ == "__main__":
    main()
