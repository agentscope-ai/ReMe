"""Deterministic query leasing for two-phase validation runs."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class QueryCasePlan:
    """One constructed case whose queries can be executed independently."""

    case_index: int
    case_id: str
    queries: tuple[Any, ...]
    context: Any = None
    owner_worker_id: int | None = None
    results: list[dict[str, Any] | None] = field(init=False)
    infrastructure_failures: list[dict[str, Any]] = field(default_factory=list)
    _pending: deque[int] = field(init=False, repr=False)
    _attempts: list[int] = field(init=False, repr=False)
    _leases: dict[int, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.results = [None] * len(self.queries)
        self._pending = deque(range(len(self.queries)))
        self._attempts = [0] * len(self.queries)

    @property
    def remaining_count(self) -> int:
        """Count queries that have not yet reached a terminal result."""

        return len(self._pending) + len(self._leases)


@dataclass(frozen=True)
class QueryLease:
    """A fenced claim for one query execution attempt."""

    token: str
    worker_id: int
    plan: QueryCasePlan
    query_index: int
    attempt: int
    selection: str

    @property
    def query(self) -> Any:
        """Return the query payload covered by this lease."""

        return self.plan.queries[self.query_index]


class QueryScheduler:
    """Assign query leases with workspace affinity and deterministic stealing.

    A case is initially owned as a whole by one worker, but ownership is an
    affinity hint rather than an exclusive lock. Idle workers may lease the
    remaining individual queries, which is what makes work stealing possible.
    """

    def __init__(self, plans: list[QueryCasePlan], *, max_retries: int) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        indexes = [plan.case_index for plan in plans]
        if len(indexes) != len(set(indexes)):
            raise ValueError("query case indexes must be unique")
        self.plans = sorted(plans, key=lambda plan: plan.case_index)
        self.max_retries = max_retries
        self._condition = asyncio.Condition()

    async def claim(self, worker_id: int, loaded_case_id: str | None) -> QueryLease | None:
        """Wait for and return the best query for one worker, or ``None`` when done."""

        async with self._condition:
            while True:
                plan, selection = self._select_plan(worker_id, loaded_case_id)
                if plan is not None:
                    query_index = plan._pending.popleft()  # pylint: disable=protected-access
                    plan._attempts[query_index] += 1  # pylint: disable=protected-access
                    token = uuid4().hex
                    plan._leases[query_index] = token  # pylint: disable=protected-access
                    if plan.owner_worker_id is None:
                        plan.owner_worker_id = worker_id
                    return QueryLease(
                        token=token,
                        worker_id=worker_id,
                        plan=plan,
                        query_index=query_index,
                        attempt=plan._attempts[query_index],  # pylint: disable=protected-access
                        selection=selection,
                    )
                if self._is_complete():
                    return None
                await self._condition.wait()

    async def complete(self, lease: QueryLease, result: dict[str, Any]) -> None:
        """Publish a terminal query result if the lease is still current."""

        async with self._condition:
            self._consume_lease(lease)
            lease.plan.results[lease.query_index] = result
            self._condition.notify_all()

    async def fail(self, lease: QueryLease, failure: dict[str, Any]) -> bool:
        """Record an infrastructure failure and requeue when retries remain.

        Returns ``True`` when the query was requeued and ``False`` when the
        supplied failure became its terminal result.
        """

        async with self._condition:
            self._consume_lease(lease)
            lease.plan.infrastructure_failures.append(failure)
            if lease.attempt <= self.max_retries:
                lease.plan._pending.appendleft(lease.query_index)  # pylint: disable=protected-access
                requeued = True
            else:
                lease.plan.results[lease.query_index] = failure
                requeued = False
            self._condition.notify_all()
            return requeued

    def _select_plan(
        self,
        worker_id: int,
        loaded_case_id: str | None,
    ) -> tuple[QueryCasePlan | None, str]:
        available = [plan for plan in self.plans if plan._pending]  # pylint: disable=protected-access
        if not available:
            return None, ""

        if loaded_case_id is not None:
            matching = [plan for plan in available if plan.case_id == loaded_case_id]
            if matching:
                owner = matching[0].owner_worker_id
                return matching[0], "affinity" if owner in (None, worker_id) else "affinity_steal"

        owned = [plan for plan in available if plan.owner_worker_id == worker_id]
        if owned:
            return self._most_remaining(owned), "owned"

        unowned = [plan for plan in available if plan.owner_worker_id is None]
        if unowned:
            return self._most_remaining(unowned), "unowned"

        return self._most_remaining(available), "steal"

    @staticmethod
    def _most_remaining(plans: list[QueryCasePlan]) -> QueryCasePlan:
        """Prefer the largest pending tail, with stable input order as a tie-breaker."""

        return min(plans, key=lambda plan: (-len(plan._pending), plan.case_index))  # pylint: disable=protected-access

    def _consume_lease(self, lease: QueryLease) -> None:
        current = lease.plan._leases.get(lease.query_index)  # pylint: disable=protected-access
        if current != lease.token:
            raise RuntimeError("stale or unknown query lease")
        del lease.plan._leases[lease.query_index]  # pylint: disable=protected-access

    def _is_complete(self) -> bool:
        return all(not plan._pending and not plan._leases for plan in self.plans)  # pylint: disable=protected-access
