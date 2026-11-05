"""Adaptive formal-validation sampling ported from dev_meta b53048e4.

Keep the source weighting, weighted ordering and fair-share query allocation.
EvaluationHistory owns persisted observations and started-validation plans.
"""

import json
import math
import random
import time
from typing import Any, Mapping

from pydantic import ConfigDict, Field

from .preparation.data.models import CaseSpec
from .preparation.spec import SpecModel


class SamplingPolicy(SpecModel):
    """Host-owned limits with the same defaults as dev_meta's validation.sampling."""

    model_config = ConfigDict(frozen=True)

    max_cases: int = Field(default=5, ge=1)
    max_queries: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    ewma_alpha: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    uniform_exploration: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)


def create_sampling_plan(
    cases: list[CaseSpec],
    state: Mapping[str, Any],
    policy: SamplingPolicy,
    *,
    eligible_query_ids: Mapping[str, list[str]] | None = None,
) -> dict:
    """Draw without replacement, without advancing state or writing an attempted plan."""
    if eligible_query_ids is not None:
        cases = [case for case in cases if case.case_id in eligible_query_ids]
    if not cases:
        raise ValueError("training-set contains no cases to sample")
    round_id = int(state.get("round", 0)) + 1
    seed = time.time_ns()
    rng = random.Random(seed)
    case_records = state.get("cases", {})
    case_weights = _mixed_weights(
        {case.case_id: case_records.get(case.case_id, {}) for case in cases},
        round_id,
        policy,
    )
    ordered_cases = _weighted_order(cases, case_weights, rng, lambda case: case.case_id)
    query_limited = policy.max_queries is not None
    remaining_queries = policy.max_queries
    selected_cases: list[str] = []
    selected_queries: dict[str, list[str]] = {}
    query_weights: dict[str, dict[str, float]] = {}
    query_records = state.get("queries", {})
    for case in ordered_cases:
        if len(selected_cases) >= policy.max_cases:
            break
        if remaining_queries is not None and remaining_queries <= 0:
            break
        query_ids = [query.query_id for query in case.queries]
        if eligible_query_ids is not None:
            allowed = set(eligible_query_ids.get(case.case_id, []))
            query_ids = [query_id for query_id in query_ids if query_id in allowed]
        if not query_ids:
            continue
        if query_limited:
            weights = _mixed_weights(
                {query_id: query_records.get(_query_key(case.case_id, query_id), {}) for query_id in query_ids},
                round_id,
                policy,
            )
            query_weights[case.case_id] = weights
            ordered_queries = _weighted_order(query_ids, weights, rng, lambda value: value)
            open_case_slots = max(policy.max_cases - len(selected_cases), 1)
            fair_share = max(math.ceil(int(remaining_queries or 0) / open_case_slots), 1)
            take = min(len(ordered_queries), fair_share, int(remaining_queries or 0))
            query_ids = ordered_queries[:take]
            if not query_ids:
                continue
            remaining_queries = int(remaining_queries or 0) - len(query_ids)
        selected_cases.append(case.case_id)
        selected_queries[case.case_id] = query_ids
    if not selected_cases:
        raise ValueError("sampling budget selected no executable case/query pairs")
    return {
        "mode": "sampled",
        "round": round_id,
        "seed": seed,
        "policy": policy.model_dump(),
        "case_ids": selected_cases,
        "query_ids": selected_queries,
        "case_draw_weights": case_weights,
        "query_draw_weights": query_weights,
        "eligibility": dict(eligible_query_ids) if eligible_query_ids is not None else None,
    }


def _observe(record: dict[str, Any], loss: float, round_id: int, alpha: float) -> None:
    previous = record.get("ewma_loss")
    record["ewma_loss"] = loss if previous is None else alpha * loss + (1.0 - alpha) * float(previous)
    record["observed_count"] += 1
    record["last_valid_score_round"] = round_id
    if loss > 0:
        record["last_error_round"] = round_id
        record["successful_observations_after_last_error"] = 0
    elif record.get("last_error_round") is not None:
        record["successful_observations_after_last_error"] += 1


def _adaptive_weight(record: Mapping[str, Any], round_id: int, policy: SamplingPolicy) -> float:
    observed = int(record.get("observed_count", 0))
    raw_risk = record.get("ewma_loss")
    risk = float(raw_risk) if isinstance(raw_risk, (int, float)) and not isinstance(raw_risk, bool) else 0.5
    last_selected = record.get("last_selected_round")
    staleness = round_id if last_selected is None else max(round_id - int(last_selected), 0)
    uncertainty = 1.0 / math.sqrt(observed + 1.0)
    last_error = record.get("last_error_round")
    successes = int(record.get("successful_observations_after_last_error", 0))
    unrechecked_error = 0.0
    if last_error is not None:
        unrechecked_error = math.log1p(max(round_id - int(last_error), 0)) / (successes + 1.0)
    priority = risk + 0.2 * math.log1p(staleness) + 0.2 * uncertainty + 0.1 * unrechecked_error
    return math.exp(min(priority / policy.temperature, 50.0))


def _mixed_weights(
    records: Mapping[str, Mapping[str, Any]],
    round_id: int,
    policy: SamplingPolicy,
) -> dict[str, float]:
    """Mix normalized adaptive weights with a true uniform exploration mass."""

    adaptive = {key: _adaptive_weight(record, round_id, policy) for key, record in records.items()}
    total = sum(adaptive.values())
    uniform = 1.0 / len(adaptive)
    return {
        key: policy.uniform_exploration * uniform + (1.0 - policy.uniform_exploration) * value / total
        for key, value in adaptive.items()
    }


def _weighted_order(values: list[Any], weights: Mapping[str, float], rng: random.Random, key: Any) -> list[Any]:
    return sorted(
        values,
        key=lambda value: rng.random() ** (1.0 / max(weights[key(value)], 1e-12)),
        reverse=True,
    )


def _empty_record() -> dict[str, Any]:
    return {
        "observed_count": 0,
        "ewma_loss": None,
        "last_selected_round": None,
        "last_valid_score_round": None,
        "last_error_round": None,
        "successful_observations_after_last_error": 0,
        "candidate_failure_count": 0,
        "infra_error_count": 0,
    }


def _query_key(case_id: str, query_id: str) -> str:
    return json.dumps([case_id, query_id], ensure_ascii=False, separators=(",", ":"))
