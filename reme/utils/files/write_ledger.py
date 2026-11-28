"""In-process record of the files written by job tools during one agent call.

A caller opens a ledger and passes its id to the agent's job tools through
``injected_job_kwargs`` under ``WRITE_LEDGER_KWARG``. The model can neither
see nor override injected arguments, so a path in the ledger was written by a
tool call made for that agent call, not by a concurrent user save.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

WRITE_LEDGER_KWARG = "_write_ledger_id"

_LEDGERS: dict[str, set[Path]] = {}


@contextmanager
def open_write_ledger() -> Iterator[tuple[str, set[Path]]]:
    """Yield a fresh ledger id and the set of resolved paths written under it."""
    ledger_id = uuid4().hex
    written: set[Path] = set()
    _LEDGERS[ledger_id] = written
    try:
        yield ledger_id, written
    finally:
        _LEDGERS.pop(ledger_id, None)


def record_write(ledger_id: object, target: Path) -> None:
    """Add ``target`` to an open ledger; unknown or closed ids are ignored."""
    if not isinstance(ledger_id, str):
        return
    written = _LEDGERS.get(ledger_id)
    if written is not None:
        written.add(Path(target).resolve())
