"""Rebuildable paired evidence; never rank candidates on different question sets."""

import math

from pydantic import Field

from reme.steps.base_step import BaseStep, Ref

from ..preparation.spec import SpecModel
from ..workspace import HomeWorkspace
from .artifacts import contained, read_json, write_json
from .inputs import entity_id
from .inspection_step import _execute, _run_path


class CompareValidationsInput(SpecModel):
    """Explicit immutable runs; detailed evidence remains in the source artifacts."""

    base_validation_path: str = Field(min_length=1)
    candidate_validation_path: str = Field(min_length=1)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load(home, path):
    run = _run_path(home, path)
    manifest = read_json(contained(run, "manifest.json"))
    summary = read_json(contained(run, "summary.json"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("mode") != "scored"
        or manifest.get("success") is not True
        or summary.get("scoring_enabled") is not True
        or any(summary.get(key, {}).get("infra_error") != 0 for key in ("case_status_counts", "query_status_counts"))
    ):
        raise ValueError("Paired comparisons require completed scored runs without infrastructure failures")
    if not manifest.get("fingerprints", {}).get("dataset") or not manifest.get("judge", {}).get("fingerprint"):
        raise ValueError("Paired comparisons require dataset and judge identity")
    queries, snapshots = {}, {}
    for case_id, ids in manifest["query_ids"].items():
        folder = contained(run, "cases/" + entity_id(case_id), directory=True)
        result = contained(folder, "queries/result.json")
        rows = read_json(result).get("queries", []) if result.is_file() else []
        indexed = {row["query_id"]: row for row in rows}
        failure_path = contained(folder, "failure.json")
        failure = read_json(failure_path) if failure_path.is_file() else {}
        build = contained(folder, "memory_construction/result.json")
        snapshots[case_id] = read_json(build).get("workspace_sha256") if build.is_file() else None
        for query_id in ids:
            row = indexed.get(query_id, {})
            score = row.get("score")
            candidate_failure = row.get("failure_kind") == "candidate_failure" or (
                not row and failure.get("failure_kind") == "candidate_failure"
            )
            # Known candidate failures count as zero; missing evidence stays unknown.
            score = float(score) if _number(score) else 0.0 if candidate_failure else None
            usage = (row.get("answer_result") or {}).get("token_usage", {})
            usage = usage if isinstance(usage, dict) else {}
            tokens = {
                model: value["total_tokens"]
                for model, value in usage.items()
                if isinstance(value, dict) and _number(value.get("total_tokens")) and value["total_tokens"] > 0
            }
            queries[(case_id, entity_id(query_id))] = {
                "score": score,
                "tokens": tokens,
                "token_models": list(usage),
                "candidate_failure": candidate_failure,
                "question_type": row.get("question_type") or "unknown",
            }
    return run, manifest, summary, queries, snapshots


def _mean(values):
    return sum(values) / len(values) if values else None


def _metrics(rows):
    scored = [row for row in rows if row["score_delta"] is not None]
    models = sorted({model for row in rows for model in row["token_models"]})
    tokens = {}
    for model in models:
        paired = [row["tokens"][model] for row in rows if model in row["tokens"]]
        tokens[model] = {
            "paired_query_count": len(paired),
            "unknown_query_count": len(rows) - len(paired),
            "base_mean": _mean([row["base"] for row in paired]),
            "candidate_mean": _mean([row["candidate"] for row in paired]),
            "mean_delta": _mean([row["delta"] for row in paired]),
        }
    return {
        "common_query_count": len(rows),
        "scored_pair_count": len(scored),
        "unknown_score_pair_count": len(rows) - len(scored),
        "base_mean_score": _mean([row["base_score"] for row in scored]),
        "candidate_mean_score": _mean([row["candidate_score"] for row in scored]),
        "mean_score_delta": _mean([row["score_delta"] for row in scored]),
        "improved_query_count": sum(row["score_delta"] > 0 for row in scored),
        "regressed_query_count": sum(row["score_delta"] < 0 for row in scored),
        "unchanged_query_count": sum(row["score_delta"] == 0 for row in scored),
        "same_snapshot_query_count": sum(row["same_construction_snapshot"] for row in rows),
        "base_candidate_failure_count": sum(row["base_candidate_failure"] for row in rows),
        "candidate_failure_count": sum(row["candidate_failure"] for row in rows),
        "query_tokens_by_model": tokens,
    }


def compare_validations(home, base_path, candidate_path):
    """Pair only identical IDs and contracts; snapshot equality is reported separately."""
    base_run, base, base_summary, before, base_snapshots = _load(home, base_path)
    candidate_run, candidate, summary, after, snapshots = _load(home, candidate_path)
    if (
        base["fingerprints"]["dataset"] != candidate["fingerprints"]["dataset"]
        or base.get("dataset_kind") != candidate.get("dataset_kind")
        or base["judge"]["fingerprint"] != candidate["judge"]["fingerprint"]
        or base_summary.get("scoring_policy") != summary.get("scoring_policy")
    ):
        raise ValueError("Cannot compare different datasets, dataset kinds or scoring policies")
    shared = sorted(before.keys() & after.keys())
    rows = []
    for case_id, query_id in shared:
        left, right = before[(case_id, query_id)], after[(case_id, query_id)]
        paired = left["score"] is not None and right["score"] is not None
        rows.append(
            {
                "case_id": case_id,
                "query_id": query_id,
                "question_type": left["question_type"],
                "base_score": left["score"],
                "candidate_score": right["score"],
                "score_delta": right["score"] - left["score"] if paired else None,
                "base_candidate_failure": left["candidate_failure"],
                "candidate_failure": right["candidate_failure"],
                "token_models": sorted(set(left["token_models"]) | set(right["token_models"])),
                "same_construction_snapshot": bool(base_snapshots[case_id])
                and base_snapshots[case_id] == snapshots[case_id],
                "tokens": {
                    model: {
                        "base": left["tokens"][model],
                        "candidate": right["tokens"][model],
                        "delta": right["tokens"][model] - left["tokens"][model],
                    }
                    for model in sorted(left["tokens"].keys() & right["tokens"].keys())
                },
            },
        )
    cases = sorted({row["case_id"] for row in rows})
    return {
        "base_validation_path": str(base_run),
        "candidate_validation_path": str(candidate_run),
        "base_commit_sha": base["commit_sha"],
        "candidate_commit_sha": candidate["commit_sha"],
        "coverage": {
            "base_query_count": len(before),
            "candidate_query_count": len(after),
            "common_case_count": len(cases),
            "base_only_query_count": len(before.keys() - after.keys()),
            "candidate_only_query_count": len(after.keys() - before.keys()),
        },
        "totals": _metrics(rows),
    }


def compact_comparison(result):
    """Keep optimizer feedback small; source paths lead to individual regressions."""
    totals = result["totals"]

    def rounded(value):
        return round(value, 4) if value is not None else None

    return {
        "base_validation_path": result["base_validation_path"],
        "candidate_validation_path": result["candidate_validation_path"],
        "coverage": {**result["coverage"], "common_query_count": totals["common_query_count"]},
        "score": {
            "base": rounded(totals["base_mean_score"]),
            "candidate": rounded(totals["candidate_mean_score"]),
            "delta": rounded(totals["mean_score_delta"]),
            "improved": totals["improved_query_count"],
            "regressed": totals["regressed_query_count"],
            "unknown_pairs": totals["unknown_score_pair_count"],
        },
        "tokens_by_model": {
            model: {
                "pairs": value["paired_query_count"],
                "base": rounded(value["base_mean"]),
                "candidate": rounded(value["candidate_mean"]),
                "delta": rounded(value["mean_delta"]),
            }
            for model, value in totals["query_tokens_by_model"].items()
        },
        "same_snapshot_queries": totals["same_snapshot_query_count"],
        "candidate_failures": {
            "base": totals["base_candidate_failure_count"],
            "candidate": totals["candidate_failure_count"],
        },
    }


def publish_comparisons(home, run):
    """Automatically save a derived table against prior compatible completed runs."""
    table, skipped = [], 0
    for manifest_path in sorted((home / "evaluations").glob("*/*/*/manifest.json")):
        other = manifest_path.parent
        if other == run:
            continue
        try:
            comparison = compact_comparison(compare_validations(home, other, run))
            comparison.pop("candidate_validation_path")
            table.append(comparison)
        except (ValueError, OSError, KeyError, TypeError):
            skipped += 1
    path = run / "paired_comparisons.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "candidate_validation_path": str(run),
            "note": "Shared questions only; null means unknown. Different snapshots do not isolate answer changes.",
            "rows": table,
            "skipped_count": skipped,
        },
        compact_leaf_containers=True,
    )
    return {"path": str(path), "comparison_count": len(table), "skipped_count": skipped}


def _compare(home, spec):
    return compact_comparison(compare_validations(home, spec.base_validation_path, spec.candidate_validation_path))


class HomeCompareValidationsStep(BaseStep):
    """Recompute paired comparisons from authoritative artifacts without model calls."""

    workspace: HomeWorkspace = Ref(HomeWorkspace, "home.workspace")

    async def execute(self):
        """Return compact paired metrics; inspect source artifacts for query evidence."""
        return await _execute(self, CompareValidationsInput, _compare)
