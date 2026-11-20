"""Shared helpers for writing normalized datasets."""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
import math
import random
from typing import Iterable

from deploy.benchmark_docker.models import validate_case_id, validate_query_artifact_id

from .models import CaseSpec, DatasetManifest, QuerySpec, SessionSpec, canonical_json_bytes, fingerprint


def select_case_ids(available: list[str], requested: Iterable[str] | None) -> list[str]:
    """Return requested IDs after checking duplicates and availability."""

    requested_ids = list(requested or [])
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Training case IDs must not contain duplicates")
    if not requested_ids:
        return available
    unknown = sorted(set(requested_ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown training case IDs: {', '.join(unknown)}")
    return requested_ids


def validate_artifact_ids(case: CaseSpec) -> None:
    """Check normalized identifiers against the sandbox and Home artifact layout."""
    try:
        validate_case_id(case.case_id)
    except ValueError as error:
        raise ValueError(f"Invalid case_id {case.case_id!r}: {error}") from error
    for query in case.queries:
        try:
            validate_query_artifact_id(query.query_id)
            if query.query_id == "result.json":
                raise ValueError("query ID conflicts with the case's query results file")
        except ValueError as error:
            raise ValueError(f"Invalid query_id {query.query_id!r} in case {case.case_id!r}: {error}") from error


def write_normalized_dataset(
    directory: Path,
    dataset: str,
    variant: str | None,
    cases: Iterable[CaseSpec],
) -> DatasetManifest:
    """Write normalized cases and their reproducibility manifest."""

    cases_directory = directory / "cases"
    cases_directory.mkdir(parents=True)
    case_ids: list[str] = []
    case_fingerprints: list[str] = []
    query_count = 0
    for index, case in enumerate(cases):
        validate_artifact_ids(case)
        if case.case_id in case_ids:
            raise ValueError(f"Duplicate normalized case ID: {case.case_id}")
        (cases_directory / f"{index:06d}.json").write_bytes(canonical_json_bytes(case) + b"\n")
        case_ids.append(case.case_id)
        case_fingerprints.append(fingerprint(case))
        query_count += len(case.queries)
    if not case_ids:
        raise ValueError("Dataset selection must contain at least one case")
    normalized_fingerprint = fingerprint(case_fingerprints)
    source_fingerprint = fingerprint(
        {
            "dataset": dataset,
            "variant": variant,
            "case_ids": case_ids,
            "normalized_fingerprint": normalized_fingerprint,
        },
    )
    manifest = DatasetManifest(
        dataset=dataset,
        source_fingerprint=source_fingerprint,
        normalized_fingerprint=normalized_fingerprint,
        case_count=len(case_ids),
        query_count=query_count,
        case_ids=case_ids,
        variant=variant,
    )
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def sessions_through_cutoff(sessions: list[SessionSpec], cutoff: datetime | None) -> list[SessionSpec]:
    """Keep eligible sessions in the caller's order."""
    if cutoff is None:
        return sessions
    return [
        session
        for session in sessions
        if (timestamp := session.first_timestamp) and datetime.fromisoformat(str(timestamp)) <= cutoff
    ]


def derive_debug_cases(
    training_root: Path,
    output: Path,
    *,
    sessions_per_case: int,
    messages_per_session: int,
    queries_per_case: int = 4,
) -> DatasetManifest:
    """Build deterministic, unscored smoke cases from normalized training data."""

    training_manifest = json.loads((training_root / "manifest.json").read_text(encoding="utf-8"))
    source_paths = sorted((training_root / "cases").glob("*.json"))
    case_count = _debug_case_count(len(source_paths))
    case_indices = _evenly_spaced_indices(len(source_paths), case_count)
    debug_cases: list[CaseSpec] = []
    for source_case_index in case_indices:
        case = CaseSpec.model_validate_json(source_paths[source_case_index].read_text(encoding="utf-8"))
        # Preserve the debug sample when none qualify. Construction reports the empty window.
        eligible_sessions = sessions_through_cutoff(case.sessions, case.construction_cutoff) or case.sessions
        sessions = []
        for index in _evenly_spaced_indices(len(eligible_sessions), sessions_per_case):
            source = eligible_sessions[index]
            sessions.append(source.model_copy(update={"messages": source.messages[:messages_per_session]}))
        debug_id = f"debug-{case.case_id}"
        debug_cases.append(
            case.model_copy(
                update={
                    "case_id": debug_id,
                    "sessions": sessions,
                    # Debug cases only smoke-test execution, not answer correctness.
                    # Keep original query/evidence metadata intentionally; referenced
                    # evidence sessions or messages may be absent after truncation.
                    "metadata": {
                        **case.metadata,
                        "debug_case": True,
                        "source_case_id": case.case_id,
                    },
                },
            ),
        )
    for case, queries in zip(debug_cases, _sample_debug_queries(debug_cases, queries_per_case)):
        case.queries = queries
    return write_normalized_dataset(output, str(training_manifest["dataset"]), "debug", debug_cases)


def _sample_debug_queries(cases: list[CaseSpec], limit: int) -> list[list[QuerySpec]]:
    """Match distinct types to case slots, then randomly fill without duplicates."""

    rng = random.Random(0)
    by_type: dict[str, dict[int, list[int]]] = {}
    slots = []
    for case_index, case in enumerate(cases):
        slots.extend((case_index, index) for index in range(min(limit, len(case.queries))))
        for query_index, query in enumerate(case.queries):
            question_type = query.metadata.get("question_type")
            if question_type:
                by_type.setdefault(str(question_type), {}).setdefault(case_index, []).append(query_index)
    options = {
        question_type: [slot for slot in slots if slot[0] in case_queries]
        for question_type, case_queries in by_type.items()
    }
    for available in options.values():
        rng.shuffle(available)
    owners: dict[tuple[int, int], str] = {}

    def assign(question_type: str, visited: set[tuple[int, int]]) -> bool:
        # Reassign earlier types when needed, so a scarce type is not crowded out.
        for slot in options[question_type]:
            if slot in visited:
                continue
            visited.add(slot)
            previous = owners.get(slot)
            if previous is None or assign(previous, visited):
                owners[slot] = question_type
                return True
        return False

    question_types = list(by_type)
    rng.shuffle(question_types)
    for question_type in question_types:
        assign(question_type, set())
    selected: list[list[int]] = [[] for _ in cases]
    for (case_index, _), question_type in owners.items():
        selected[case_index].append(rng.choice(by_type[question_type][case_index]))
    result = []
    for case, indices in zip(cases, selected):
        remaining = [index for index in range(len(case.queries)) if index not in indices]
        indices.extend(rng.sample(remaining, min(limit, len(case.queries)) - len(indices)))
        rng.shuffle(indices)
        result.append([case.queries[index] for index in indices])
    return result


def _debug_case_count(training_case_count: int) -> int:
    """Use ten percent of training cases, with a four-case floor when possible."""

    return min(training_case_count, max(4, math.ceil(training_case_count * 0.1)))


def _evenly_spaced_indices(size: int, limit: int) -> list[int]:
    """Choose stable beginning/middle/end positions without duplicates."""

    if size <= limit:
        return list(range(size))
    if limit == 1:
        return [0]
    return sorted({round(index * (size - 1) / (limit - 1)) for index in range(limit)})
