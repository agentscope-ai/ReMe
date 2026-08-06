"""Shared helpers for writing normalized Meta-ReMe datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from models import CaseSpec, DatasetManifest, canonical_json_bytes, fingerprint


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
        (cases_directory / f"{index:06d}.json").write_bytes(canonical_json_bytes(case) + b"\n")
        case_ids.append(case.case_id)
        case_fingerprints.append(fingerprint(case))
        query_count += len(case.queries)
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
    )
    (directory / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest
