"""Host-owned ancestor selection and 60/40 construction replay policy."""

from pathlib import Path
import random
import subprocess
import tarfile

from reme.utils.async_utils import complete_in_thread as blocking

from ..code.repository import current_branch, validate_candidate, validate_repository
from ..code.snapshot import tree
from ..preparation.code import git
from ..preparation.data.models import fingerprint
from .artifacts import contained, file_sha256, read_json
from .inputs import (
    EvaluationSpec,
    load_reuse_source,
    reuse_inputs,
    select_cases,
    validate_replay_answers,
    validate_snapshot,
)

AUTO_MEMORY_FILENAMES = {
    "auto_memory.py",
    "auto_memory.yaml",
    "auto_memory.yml",
    "auto-memory.py",
    "auto-memory.yaml",
    "auto-memory.yml",
}


def construction_fingerprints(record, dataset, spec, environment):
    """Match the runtime, config, plugins, image and construction model identity."""
    return {
        "dataset": dataset.normalized_fingerprint,
        "construction_config": fingerprint(
            {
                "runtime": record["code"]["runtime_fingerprint"],
                "config": record["code"]["config"],
                "plugins": record["code"]["plugins"],
                "image": spec.base_image,
                "models": {
                    k: v for k, v in environment.items() if "MODEL" in k and "KEY" not in k and "JUDGE" not in k
                },
            },
        ),
    }


def verify_candidate(home, commit, branch, *, checkout_branch=None):
    """Check the prepared checkout while the caller holds the workspace lock."""
    repo, record, current = validate_candidate(home)
    if current == record["code"]["initial_commit"]:
        actual_branch = "init" if checkout_branch is None else git(repo, "branch", "--show-current")
    else:
        actual_branch = current_branch(repo)
    expected_branch = branch if checkout_branch is None else checkout_branch
    if (current, actual_branch) != (commit, expected_branch):
        raise ValueError("Candidate changed during equivalence review; retry evaluation on the reviewed commit")
    return repo, record


def reuse_context(home, spec, cases, fingerprints, commit, branch, *, replay_selection=False, targeted=False):
    """Find compatible ancestors from saved metadata without opening any archives."""
    repo, record = verify_candidate(home, commit, branch)
    if commit == record["code"]["initial_commit"]:
        return None
    parent = git(repo, "rev-parse", f"{commit}^1")
    paired = replay_selection and parent != record["code"]["initial_commit"]
    identity = {"candidate_commit": commit, "branch_name": branch, "parent_commit": parent}
    ancestors = set(git(repo, "rev-list", commit).splitlines()) - {commit}
    distances = {}
    candidates = []
    for path in sorted((home / "evaluations").glob("*/*/*/manifest.json")):
        try:
            manifest = read_json(contained(home, path))
            if not isinstance(manifest, dict):
                continue
            base = manifest.get("commit_sha")
            if (
                base not in ancestors
                or manifest.get("mode") != "scored"
                or manifest.get("dataset_kind") != "training-set"
                or manifest.get("status") != "completed"
                or manifest.get("success") is False
            ):
                continue
            source_fingerprints = manifest.get("fingerprints")
            if not isinstance(source_fingerprints, dict) or any(
                source_fingerprints.get(key) != fingerprints[key] for key in ("dataset", "construction_config")
            ):
                continue
            summary = read_json(contained(home, path.parent / "summary.json"))
            if not isinstance(summary, dict):
                continue
            counts = [summary.get(key, {}) for key in ("case_status_counts", "query_status_counts")]
            if any(not isinstance(value, dict) or value.get("infra_error", 0) for value in counts):
                continue
            if base not in distances:
                distances[base] = int(git(repo, "rev-list", "--count", f"{base}..{commit}"))
            case_ids = _reusable_case_hints(summary.get("cases")) if targeted else ()
            candidates.append((distances[base], str(path.parent), base, case_ids))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    candidates.sort(key=lambda entry: entry[:3])
    # A summary only nominates a source. Archive and answer checks happen after sampling.
    candidate_sources = {}
    for _, path, _, case_ids in candidates:
        for case_id in case_ids:
            candidate_sources.setdefault(case_id, path)
    return {
        "context": identity,
        "spec": spec,
        "cases": cases,
        "fingerprints": fingerprints,
        "sources": [(path, base) for _, path, base, _ in candidates],
        "candidate_sources": candidate_sources if targeted else None,
        "paired_replay": paired,
    }


def _reusable_case_hints(rows):
    """Use the summary to nominate cases with a full build and successful answers."""
    if not isinstance(rows, dict):
        return []
    return [
        case_id
        for case_id, row in rows.items()
        if isinstance(row, dict)
        and isinstance(build := row.get("construction"), dict)
        and build.get("status") == "success"
        and row.get("queries")
        and not row.get("query_error_count")
    ]


def reuse_response(reuse):
    """Expand derived tool response fields in their usual order."""
    response = {}
    for key, value in reuse.items():
        if key == "probability_selected":
            response["probability_scope"] = "all_cases"
        response[key] = value
        if key == "source_validation":
            report = reuse.get("source_reviews", {}).get(value, {}).get("equivalence_report")
            if report is not None:
                response["equivalence_report"] = report
            elif reuse["case_sources"]:
                raise ValueError(f"Missing construction-equivalence report for replay: {value}")
    response["reuse_reasons"] = {case_id: "60_percent_policy" for case_id in reuse["case_sources"]}
    return response


def _replay_selection(spec, cases, manifest):
    """Keep one historical question set intact, within the current sampling limits."""
    case_ids, query_ids = manifest["case_ids"], manifest["query_ids"]
    if not case_ids or set(query_ids) != set(case_ids):
        return None
    policy = spec.sampling_policy
    if len(case_ids) > policy.max_cases or (
        policy.max_queries is not None and sum(len(ids) for ids in query_ids.values()) > policy.max_queries
    ):
        return None
    selection = {"case_ids": case_ids, "query_ids": query_ids}
    selection_spec = type(spec).model_validate(
        {**spec.model_dump(), "sampling": False, "target_queries": None, **selection},
    )
    selected = select_cases({case.case_id: case for case in cases}, selection_spec.case_ids, selection_spec.query_ids)
    return selected, selection


def prepare_source(home, spec, cases, fingerprints, context, *, paired=False):
    """Validate candidate snapshots, requiring every case for a paired replay."""
    path = context["source_validation"]
    source_spec = spec.model_copy(update={"reuse_construction_from": path})
    try:
        source, manifest = load_reuse_source(home, path, fingerprints)
        if manifest["commit_sha"] != context["base_commit"]:
            return None
        if paired:
            replay = _replay_selection(spec, cases, manifest)
            if replay is None:
                return None
            cases, selection = replay
    except (OSError, ValueError, KeyError, TypeError):
        return None
    reusable = []
    for case in cases:
        try:
            validate_snapshot(source, case.case_id, spec.max_artifact_bytes)
            validate_replay_answers(source, case.case_id)
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError):
            if paired:
                return None
            continue
        reusable.append(case)
    if not reusable:
        return None
    prepared = {"context": context, "spec": source_spec, "cases": reusable, "fingerprints": fingerprints}
    if paired:
        prepared["replay_selection"] = selection
    return prepared


def case_reuse_inputs(home, spec, cases, fingerprints, sources):
    """Resolve a frozen per-case source map without substituting another snapshot."""
    by_id = {case.case_id: case for case in cases}
    if set(sources) - by_id.keys():
        raise ValueError("Construction sources contain unselected cases")
    groups = {}
    for case_id, source in sources.items():
        groups.setdefault(source["result_dir"], []).append(by_id[case_id])
    resolved, archives = {}, {}
    for path, selected in groups.items():
        source, snapshots = reuse_inputs(
            home,
            spec.model_copy(update={"reuse_construction_from": path}),
            selected,
            fingerprints,
        )
        for case in selected:
            if source["commit_sha"] != sources[case.case_id]["commit_sha"]:
                raise ValueError("Construction source commit changed")
            resolved[case.case_id] = source
        archives.update(snapshots)
    return resolved, archives


def verify_case_reuse(home, run, manifest, commit, cases):
    """Accept mixed evidence only when every reused case has host-reviewed provenance."""
    sources = manifest["construction_sources"]
    selected = manifest.get("case_ids", [])
    reuse = manifest.get("automatic_construction_reuse", {})
    expected_mode = "query_only_replay" if set(sources) == set(selected) else "mixed_validation"
    if not all(
        (
            bool(sources),
            not set(sources) - set(selected),
            manifest.get("execution_mode") == expected_mode,
            reuse.get("selected") is True,
            reuse.get("policy") == "automatic_60_percent",
            reuse.get("candidate_commit") == commit,
            reuse.get("branch_name") == manifest.get("branch_name"),
            reuse.get("case_sources") == sources,
        ),
    ):
        raise ValueError("Reused evidence requires host-selected per-case construction sources")
    targets = set(manifest.get("target_selection", {}).get("retained_case_ids", []))
    if targets != set(reuse.get("target_case_ids", [])) or reuse.get("probability_selected") is not True:
        raise ValueError("Reused cases require matching targets and a successful host probability draw")
    for case_id, source in sources.items():
        group = reuse.get("source_reviews", {}).get(source["result_dir"], {})
        report = group.get("equivalence_report", {})
        if not all(
            (
                case_id in group.get("case_ids", []),
                report.get("decision") == "construction_unchanged",
                group.get("candidate_commit") == commit,
                report.get("candidate_commit") == commit,
                group.get("base_commit") == source["commit_sha"],
                report.get("base_commit") == source["commit_sha"],
                group.get("source_validation") == source["result_dir"],
                report.get("source_validation") == source["result_dir"],
            ),
        ):
            raise ValueError(f"Case {case_id} lacks matching construction-equivalence evidence")
    spec = EvaluationSpec.model_validate(manifest.get("options", {}))
    cases = [case for case in cases if case.case_id in sources]
    _, archives = case_reuse_inputs(home, spec, cases, manifest["fingerprints"], sources)
    for case_id in selected:
        folder = contained(run, f"cases/{case_id}/memory_construction", directory=True)
        result = read_json(folder / "result.json")
        if case_id not in sources:
            if result.get("status") == "reused":
                raise ValueError(f"Case {case_id} has an undeclared reused snapshot")
            continue
        if (
            result.get("success") is not True
            or result.get("status") != "reused"
            or result.get("source") != sources[case_id]
            or result.get("workspace_sha256") != archives[case_id][1]
            or file_sha256(folder / "reme_workspace.tar.gz") != archives[case_id][1]
        ):
            raise ValueError(f"Case {case_id} has mismatched construction snapshot provenance")


def changed_auto_memory_files(home, context):
    """Preserve dev_meta's conservative override for substantive auto-memory changes."""
    repo, _ = validate_repository(home)
    base, candidate = context["base_commit"], context["candidate_commit"]
    old, new = dict(tree(repo, base)), dict(tree(repo, candidate))
    changed = []
    for name in sorted(old.keys() | new.keys()):
        if Path(name).name not in AUTO_MEMORY_FILENAMES or old.get(name) == new.get(name):
            continue
        try:
            git(
                repo,
                "diff",
                "--quiet",
                "--no-ext-diff",
                "--no-textconv",
                "--ignore-all-space",
                "--ignore-blank-lines",
                base,
                candidate,
                "--",
                ":(literal)" + name,
            )
        except subprocess.CalledProcessError as error:
            if error.returncode != 1:
                raise
            changed.append(name)
    return changed


def select_query_only():
    """Draw once after an unchanged decision; retain the draw in host evidence."""
    draw = random.SystemRandom().random()
    return {"probability": 0.6, "draw": draw, "selected": draw < 0.6}


async def _review_source(wrapper, home, prepared, *, memory_limit_mib=None):
    """Review one source without drawing a separate probability for each case."""
    from ..agent.construction_equivalence import review_construction
    from ..agent.workflow_common import save_record

    context = prepared["context"]
    changed = await blocking(changed_auto_memory_files, home, context)
    if changed:
        report = {
            **context,
            "decision": "construction_changed",
            "reason": "Substantive auto-memory changes require rebuilding: " + ", ".join(changed),
        }
        report["review_path"] = await blocking(save_record, home, "construction-equivalence", report)
    else:
        report = await review_construction(
            wrapper,
            home,
            context,
            prepared["spec"],
            prepared["cases"],
            prepared["fingerprints"],
            memory_limit_mib=memory_limit_mib,
        )
    return report


async def automatic_reuse(wrapper, home, prepared, *, memory_limit_mib=None, cases=None):
    """Review the selected snapshots, then draw once for all authorized cases."""
    if prepared is None:
        return None
    context = prepared["context"]
    requested_cases = prepared["cases"] if cases is None else cases
    candidate_sources = prepared["candidate_sources"]
    selection = {
        **context,
        "policy": "automatic_60_percent",
        "selected": False,
        "source_reviews": {},
        "case_sources": {},
        "target_case_ids": [],
        "probability_selected": None,
    }
    pending = {case.case_id: case for case in requested_cases}
    approved = {}
    for path, base in prepared["sources"]:
        if not pending:
            break
        selected = list(pending.values())
        if candidate_sources is not None:
            selected = [case for case in selected if candidate_sources.get(case.case_id) == path]
            # Budget one candidate snapshot per case. Failed reviews or stale
            # hints fall back to construction without checking more archives.
            for case in selected:
                pending.pop(case.case_id)
        if not selected:
            continue
        group = await blocking(
            prepare_source,
            home,
            prepared["spec"],
            selected,
            prepared["fingerprints"],
            {**context, "base_commit": base, "source_validation": path},
            paired=prepared["paired_replay"],
        )
        if group is None:
            continue
        report = await _review_source(wrapper, home, group, memory_limit_mib=memory_limit_mib)
        group_context = group["context"]
        if "source_validation" not in selection:
            selection.update(group_context)
        selection["source_reviews"][path] = {
            **group_context,
            "case_ids": [case.case_id for case in group["cases"]],
            "equivalence_report": report,
        }
        if report["decision"] == "construction_unchanged":
            if not approved:
                selection.update(group_context)
            for case in group["cases"]:
                approved[case.case_id] = {"result_dir": path, "commit_sha": group_context["base_commit"]}
                pending.pop(case.case_id, None)
            if prepared["paired_replay"]:
                selection["replay_selection"] = group["replay_selection"]
                break
    if approved:
        draw = select_query_only()
        selection.update({key: value for key, value in draw.items() if key != "selected"})
        selection["probability_selected"] = draw["selected"]
        if draw["selected"]:
            selection["case_sources"].update(approved)
        selection["selected"] = bool(selection["case_sources"])
        selection["reason"] = (
            "query_only_selected_by_60_percent_policy"
            if selection["selected"]
            else "full_validation_selected_by_60_percent_policy"
        )
    else:
        selection["reason"] = "construction_not_equivalent" if selection["source_reviews"] else "no_reusable_ancestor"
    rebuild_reason = (
        "no_reusable_ancestor" if selection["reason"] == "no_reusable_ancestor" else "no_compatible_reviewed_snapshot"
    )
    selection["rebuild_reasons"] = {
        case.case_id: (
            "full_validation_selected_by_60_percent_policy"
            if case.case_id in approved and selection["probability_selected"] is False
            else rebuild_reason
        )
        for case in requested_cases
        if case.case_id not in selection["case_sources"]
    }
    return selection
