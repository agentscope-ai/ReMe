"""Finish owned-resource cleanup before propagating caller cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable


def format_error(error: BaseException) -> str:
    """Include cleanup notes when an exception becomes a result string."""
    return "; ".join([f"{type(error).__name__}: {error}", *getattr(error, "__notes__", ())])


async def finish_cleanup(
    operations: Iterable[Awaitable[None]],
) -> tuple[list[None | BaseException], asyncio.CancelledError | None]:
    """Collect all failures, shielding teardown from repeated cancellation.

    The caller must propagate the returned cancellation after reporting errors.
    Individual cleanup operations may still fail or cancel themselves.
    """

    async def collect(operation: Awaitable[None]) -> None | BaseException:
        try:
            await operation
        except BaseException as exc:
            # gather(return_exceptions=True) replaces CancelledError and loses
            # its notes. Keep the original exception for cleanup diagnostics.
            return exc
        return None

    cleanup = asyncio.gather(*(collect(operation) for operation in operations))
    cancelled = None
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError as exc:
            cancelled = exc
    return cleanup.result(), cancelled


async def cleanup_on_error(operations: Iterable[tuple[str, Awaitable[None]]], error: BaseException) -> None:
    """Attach teardown failures to the primary error before propagating cancellation."""
    operations = list(operations)
    outcomes, cancelled = await finish_cleanup(operation for _, operation in operations)
    for (name, _), outcome in zip(operations, outcomes):
        if isinstance(outcome, BaseException):
            error.add_note(f"{name} close failed: {format_error(outcome)}")
    if cancelled is not None and not isinstance(error, asyncio.CancelledError):
        for note in getattr(error, "__notes__", ()):
            cancelled.add_note(note)
        raise cancelled from error
