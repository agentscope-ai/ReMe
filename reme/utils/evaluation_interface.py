"""Read-only evaluation helpers for application job execution statistics.

These helpers take before/after snapshots of application-lifetime counters.
They are intentionally not thread-safe request attribution: overlapping calls
in the same Application contribute to each other's deltas. They are intended
for the benchmark utilities, where each tracked evaluation runs without other
work sharing its Application instance.
"""

from typing import TYPE_CHECKING

from .counter import global_counter_get, global_counter_get_all

if TYPE_CHECKING:
    from ..components.application_context import ApplicationContext


_TOKEN_METRICS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
)


def check_job_count(job_name: str, app_context: "ApplicationContext") -> int:
    """Return the application-lifetime execution count for a registered job.

    ``app_context`` scopes the lookup because ReMe does not maintain a global
    current Application instance. Unknown job names use the same ``KeyError``
    contract as :meth:`Application.run_job`.
    """
    if job_name not in app_context.jobs:
        raise KeyError(f"Job '{job_name}' not found")
    return global_counter_get(app_context.metadata, ["__job_counter", job_name])


class JobCountTracker:
    """Measure registered job calls made while this context is active.

    Not thread-safe for per-request attribution; see the module docstring.
    """

    def __init__(self, job_names: list[str], app_context: "ApplicationContext") -> None:
        self.job_names = list(dict.fromkeys(job_names))
        self.app_context = app_context
        self._start_counts: dict[str, int] = {}
        self.counts: dict[str, int] = {}

    def __enter__(self) -> dict[str, int]:
        self._start_counts = {name: check_job_count(name, self.app_context) for name in self.job_names}
        return self.counts

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.counts.update(
            {
                name: check_job_count(name, self.app_context) - start_count
                for name, start_count in self._start_counts.items()
            },
        )
        return False


def track_job_counts(job_names: list[str], app_context: "ApplicationContext") -> JobCountTracker:
    """Return a context manager that reports call deltas for ``job_names``.

    Example:

    .. code-block:: python

        with track_job_counts(["search"], app.context) as counts:
            await app.run_job("agentic_answer", query="...")
        assert counts == {"search": 2}
    """
    return JobCountTracker(job_names, app_context)


def check_agent_token_count(
    agent_name: str,
    app_context: "ApplicationContext",
    metric: str = "total_tokens",
) -> int:
    """Return one application-lifetime token metric for an agent wrapper.

    The agent name is the configured ``agent_wrapper`` component name (for
    example ``"bench"``), and ``metric`` is one leaf in ReMe's token counter
    tree, such as ``input_tokens`` or ``total_tokens``.
    """
    return global_counter_get(app_context.metadata, ["__token_counter", agent_name, metric])


def check_agent_token_usage(agent_name: str, app_context: "ApplicationContext") -> dict[str, int | None]:
    """Return all token metrics currently accumulated for one agent wrapper.

    A metric remains ``None`` until the backend has reported it. This keeps
    unavailable usage distinct from zero.
    """
    tree = global_counter_get_all(app_context.metadata, ["__token_counter", agent_name])
    children = tree.get("children", {}) if tree is not None else {}
    usage: dict[str, int | None] = {}
    for metric in _TOKEN_METRICS:
        node = children.get(metric)
        usage[metric] = node["value"] if node is not None else None
    return usage


class AgentTokenCountTracker:
    """Measure one token metric for agent wrappers during a context block.

    Not thread-safe for per-request attribution; see the module docstring.
    """

    def __init__(
        self,
        agent_names: list[str],
        app_context: "ApplicationContext",
        metric: str = "total_tokens",
    ) -> None:
        self.agent_names = list(dict.fromkeys(agent_names))
        self.app_context = app_context
        self.metric = metric
        self._start_counts: dict[str, int] = {}
        self.counts: dict[str, int] = {}

    def __enter__(self) -> dict[str, int]:
        self._start_counts = {
            name: check_agent_token_count(name, self.app_context, self.metric) for name in self.agent_names
        }
        return self.counts

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.counts.update(
            {
                name: check_agent_token_count(name, self.app_context, self.metric) - start_count
                for name, start_count in self._start_counts.items()
            },
        )
        return False


def track_agent_token_counts(
    agent_names: list[str],
    app_context: "ApplicationContext",
    metric: str = "total_tokens",
) -> AgentTokenCountTracker:
    """Return a context manager that reports agent token deltas.

    Example:

    .. code-block:: python

        with track_agent_token_counts(["bench"], app.context) as counts:
            await app.run_job("agentic_answer", query="...")
        assert counts["bench"] > 0
    """
    return AgentTokenCountTracker(agent_names, app_context, metric)


class AgentTokenUsageTracker:
    """Measure all token metrics for agent wrappers during a context block.

    Not thread-safe for per-request attribution; see the module docstring.
    """

    def __init__(self, agent_names: list[str], app_context: "ApplicationContext") -> None:
        self.agent_names = list(dict.fromkeys(agent_names))
        self.app_context = app_context
        self._start_usage: dict[str, dict[str, int | None]] = {}
        self.usages: dict[str, dict[str, int | None]] = {}

    def __enter__(self) -> dict[str, dict[str, int | None]]:
        self._start_usage = {name: check_agent_token_usage(name, self.app_context) for name in self.agent_names}
        return self.usages

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for name in self.agent_names:
            end_usage = check_agent_token_usage(name, self.app_context)
            delta: dict[str, int | None] = {}
            for metric in _TOKEN_METRICS:
                current = end_usage[metric]
                start = self._start_usage[name][metric]
                delta[metric] = None if current is None else current - (start or 0)
            self.usages[name] = delta
        return False


def track_agent_token_usage(
    agent_names: list[str],
    app_context: "ApplicationContext",
) -> AgentTokenUsageTracker:
    """Return a context manager that reports full per-agent token usage deltas."""
    return AgentTokenUsageTracker(agent_names, app_context)
