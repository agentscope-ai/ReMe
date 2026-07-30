"""Read-only evaluation helpers for application job execution statistics."""

from typing import TYPE_CHECKING

from ..components.job import BackgroundJob, BaseJob, CronJob, StreamJob
from .counter import global_counter_get

if TYPE_CHECKING:
    from ..components.application_context import ApplicationContext


_JOB_ENTRY_CLASSES = {CronJob, StreamJob, BackgroundJob, BaseJob}


def check_job_count(job_name: str, app_context: "ApplicationContext") -> int:
    """Return the application-lifetime execution count for a registered job.

    ``app_context`` scopes the lookup because ReMe does not maintain a global
    current Application instance. Unknown job names use the same ``KeyError``
    contract as :meth:`Application.run_job`.
    """
    job = app_context.jobs.get(job_name)
    if job is None:
        raise KeyError(f"Job '{job_name}' not found")

    entry_class = next((cls for cls in type(job).mro() if cls in _JOB_ENTRY_CLASSES), None)
    if entry_class is None:
        raise TypeError(f"Job '{job_name}' does not inherit from a supported job implementation")
    # pylint: disable-next=protected-access
    return global_counter_get(app_context.metadata, job._counter_key(entry_class))
