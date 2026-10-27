"""Shared helpers for writing normalized Meta-ReMe datasets."""

from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
import math
from typing import Iterable

from .models import CaseSpec, DatasetManifest, canonical_json_bytes, fingerprint


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


def derive_debug_cases(
    training_root: Path,
    output: Path,
    *,
    sessions_per_case: int,
    messages_per_session: int,
) -> DatasetManifest:
    """Build deterministic, unscored smoke cases from normalized training data."""

    training_manifest = json.loads((training_root / "manifest.json").read_text(encoding="utf-8"))
    source_cases = [
        CaseSpec.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((training_root / "cases").glob("*.json"))
    ]
    case_count = _debug_case_count(len(source_cases))
    case_indices = _evenly_spaced_indices(len(source_cases), case_count)
    selected = [source_cases[index] for index in case_indices]
    debug_cases: list[CaseSpec] = []
    derivations: list[dict[str, object]] = []
    for source_case_index, case in zip(case_indices, selected, strict=True):
        eligible_indices = _eligible_session_indices(case)
        selected_positions = _evenly_spaced_indices(len(eligible_indices), sessions_per_case)
        session_indices = [eligible_indices[position] for position in selected_positions]
        sessions = []
        for index in session_indices:
            source = case.sessions[index]
            sessions.append(source.model_copy(update={"messages": source.messages[:messages_per_session]}))
        debug_id = f"debug-{case.case_id}"
        debug_cases.append(
            case.model_copy(
                update={
                    "case_id": debug_id,
                    "sessions": sessions,
                    "queries": case.queries[:1],
                    "metadata": {
                        **case.metadata,
                        "debug_case": True,
                        "source_case_id": case.case_id,
                    },
                },
            ),
        )
        derivations.append(
            {
                "debug_case_id": debug_id,
                "source_case_id": case.case_id,
                "source_case_index": source_case_index,
                "session_indices": session_indices,
                "messages_per_session": messages_per_session,
            },
        )
    manifest = write_normalized_dataset(output, str(training_manifest["dataset"]), "debug", debug_cases)
    (output / "derivation.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "training_fingerprint": training_manifest["normalized_fingerprint"],
                "rules": {
                    "case_selection_ratio": 0.1,
                    "minimum_case_count": 2,
                    "derived_case_count": case_count,
                    "sessions_per_case": sessions_per_case,
                    "messages_per_session": messages_per_session,
                    "case_selection": "evenly_spaced_in_normalized_order",
                    "session_selection": "evenly_spaced",
                    "message_selection": "prefix",
                },
                "cases": derivations,
            },
        )
        + b"\n",
    )
    return manifest


def _debug_case_count(training_case_count: int) -> int:
    """Use ten percent of training cases, with a two-case floor when possible."""

    return min(training_case_count, max(2, math.ceil(training_case_count * 0.1)))


def _evenly_spaced_indices(size: int, limit: int) -> list[int]:
    """Choose stable beginning/middle/end positions without duplicates."""

    if size <= limit:
        return list(range(size))
    if limit == 1:
        return [0]
    return sorted({round(index * (size - 1) / (limit - 1)) for index in range(limit)})


def _eligible_session_indices(case: CaseSpec) -> list[int]:
    """Mirror LongMemEval's query-time cutoff so a debug case remains buildable."""

    if case.metadata.get("dataset") != "longmemeval" or not case.queries:
        return list(range(len(case.sessions)))
    question_date = case.queries[0].metadata.get("question_date")
    if not isinstance(question_date, str) or not question_date:
        return list(range(len(case.sessions)))
    query_time = datetime.strptime(question_date, "%Y/%m/%d (%a) %H:%M")
    eligible = []
    for index, session in enumerate(case.sessions):
        timestamps = [message.get("created_at") for message in session.messages if message.get("created_at")]
        if timestamps and datetime.fromisoformat(str(timestamps[0])) <= query_time:
            eligible.append(index)
    return eligible or list(range(len(case.sessions)))
