"""Dataset-specific normalization for Meta-ReMe training data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from data_preparation import beam, lme
from data_preparation.basic import write_normalized_dataset
from models import DatasetManifest


def prepare_dataset(
    directory: Path,
    dataset: str,
    source: Path,
    train_case_ids: Iterable[str] | None = None,
    variant: str | None = None,
) -> DatasetManifest:
    """Normalize the selected training cases into ``directory``."""

    if dataset == "beam":
        selected_variant = variant or "1M"
        if selected_variant not in beam.VARIANTS:
            raise ValueError(f"Unsupported BEAM variant: {selected_variant}")
        selected = beam.selected_case_ids(source, selected_variant, train_case_ids)
        cases = beam.iter_cases(source, selected_variant, selected)
        return write_normalized_dataset(directory, dataset, selected_variant, cases)
    if dataset == "longmemeval":
        if variant is not None:
            raise ValueError("LongMemEval does not accept a dataset variant")
        return write_normalized_dataset(directory, dataset, None, lme.iter_cases(source, train_case_ids))
    raise ValueError(f"Unsupported dataset: {dataset}")


__all__ = ["prepare_dataset"]
