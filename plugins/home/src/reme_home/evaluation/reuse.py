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
from .inputs import load_inputs, reuse_inputs
from .repetition import previous_candidate_result

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


def reuse_context(home, spec, environment, cases=None, *, replay_selection=False):
    """Find ancestor sources; an init parent still requires a fresh budgeted sample."""
    repo, record, commit = validate_candidate(home)
    identity = {"candidate_commit": commit, "branch_name": current_branch(repo)}
    previous = previous_candidate_result(home, commit)
    if previous is not None:
        saved = read_json(contained(home, previous / "manifest.json")).get("automatic_construction_reuse")
        # A managed branch may move to the same failed commit. Retain its
        # commit-pinned review/draw while preserving the existing retry contract.
        return {"recovery": {**saved, **identity}} if saved is not None else None
    if commit == record["code"]["initial_commit"]:
        return None
    parent = git(repo, "rev-parse", f"{commit}^1")
    paired = replay_selection and parent != record["code"]["initial_commit"]
    identity["parent_commit"] = parent
    _, dataset, loaded = load_inputs(home, spec)
    cases = loaded if cases is None else cases
    fingerprints = construction_fingerprints(record, dataset, spec, environment)
    ancestors = set(git(repo, "rev-list", commit).splitlines()) - {commit}
    candidates = []
    for path in sorted((home / "evaluations").glob("*/*/*/manifest.json")):
        try:
            manifest = read_json(contained(home, path))
            base = manifest.get("commit_sha")
            if base not in ancestors:
                continue
            summary = read_json(contained(home, path.parent / "summary.json"))
            if (
                manifest.get("mode") != "scored"
                or manifest.get("success") is False
                or summary.get("case_status_counts", {}).get("infra_error", 0)
                or summary.get("query_status_counts", {}).get("infra_error", 0)
            ):
                continue
            distance = int(git(repo, "rev-list", "--count", f"{base}..{commit}"))
            candidates.append((distance, str(path.parent), base))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    candidates.sort()
    # Check older sources lazily, only for cases not authorized by a nearer source.
    for index, (_, source, base) in enumerate(candidates):
        prepare = prepare_replay_source if paired else prepare_source
        prepared = prepare(
            home,
            spec,
            cases,
            fingerprints,
            {**identity, "base_commit": base, "source_validation": source},
        )
        if prepared is not None:
            return {
                **prepared,
                "remaining_sources": candidates[index + 1 :],
                "requested_cases": cases,
                "paired_replay": paired,
            }
    return {
        **identity,
        "selected": False,
        "reason": "no_reusable_ancestor",
        "policy": "automatic_60_percent",
        "rebuild_reasons": {case.case_id: "no_reusable_ancestor" for case in cases},
    }


def prepare_replay_source(home, spec, cases, fingerprints, context):
    """Keep one historical question set intact, within the current sampling limits."""
    try:
        manifest = read_json(contained(home, context["source_validation"] + "/manifest.json"))
        case_ids, query_ids = manifest["case_ids"], manifest["query_ids"]
        if not case_ids or set(query_ids) != set(case_ids):
            return None
        policy = spec.sampling_policy
        if len(case_ids) > policy.max_cases or (
            policy.max_queries is not None and sum(len(ids) for ids in query_ids.values()) > policy.max_queries
        ):
            return None
        selection_spec = spec.model_copy(
            update={"sampling": False, "target_queries": None, "case_ids": case_ids, "query_ids": query_ids},
        )
        # Validate saved IDs, including uniqueness, instead of silently dropping pairs.
        selection_spec = type(spec).model_validate(selection_spec.model_dump())
        _, _, selected = load_inputs(home, selection_spec)
        if set(case_ids) - {case.case_id for case in cases}:
            return None
        prepared = prepare_source(home, spec, selected, fingerprints, context)
        if prepared is None or len(prepared["cases"]) != len(selected):
            return None
        prepared["replay_selection"] = {"case_ids": case_ids, "query_ids": query_ids}
        return prepared
    except (OSError, ValueError, KeyError, TypeError):
        return None


def prepare_source(home, spec, cases, fingerprints, context):
    """An invalid or missing case snapshot does not disqualify the other cases."""
    source_spec = spec.model_copy(update={"reuse_construction_from": context["source_validation"]})
    reusable = []
    for case in cases:
        try:
            reuse_inputs(home, source_spec, [case], fingerprints)
            source = contained(home, context["source_validation"], directory=True)
            rows = read_json(contained(source, f"cases/{case.case_id}/queries/result.json"))
            queries = rows.get("queries")
            if not queries or not all(
                isinstance(query, dict)
                and query.get("error") is None
                and isinstance(query.get("answer"), str)
                and query["answer"].strip()
                for query in queries
            ):
                continue
            reusable.append(case)
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError):
            continue
    if not reusable:
        return None
    return {"context": context, "spec": source_spec, "cases": reusable, "fingerprints": fingerprints}


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


def verify_case_reuse(home, run, manifest, commit):
    """Accept mixed evidence only when every reused case has host-reviewed provenance."""
    from .inputs import EvaluationSpec

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
    if "probability_selected" in reuse:
        targets = set(manifest.get("target_selection", {}).get("retained_case_ids", []))
        if targets != set(reuse.get("target_case_ids", [])) or (
            set(sources) - targets and reuse["probability_selected"] is not True
        ):
            raise ValueError("Reused cases require target priority or a successful host probability draw")
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
    _, _, cases = load_inputs(home, spec)
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


async def automatic_reuse(wrapper, home, prepared, *, memory_limit_mib=None, target_case_ids=()):
    """Always reuse authorized targets; draw once for other authorized cases."""
    if prepared is None:
        return None
    if "recovery" in prepared:
        return prepared["recovery"]
    if "context" not in prepared:
        return prepared
    context = prepared["context"]
    selection = {
        **context,
        "policy": "automatic_60_percent",
        "selected": False,
        "source_reviews": {},
        "case_sources": {},
        "target_case_ids": sorted(target_case_ids),
        "probability_scope": "untargeted_cases" if target_case_ids else "all_cases",
        "probability_selected": None,
    }
    pending = {case.case_id: case for case in prepared.get("requested_cases", prepared["cases"])}
    approved = {}
    sources = [None, *prepared.get("remaining_sources", [])]
    for source in sources:
        if not pending:
            break
        if source is None:
            group = prepared
        else:
            _, path, base = source
            group = await blocking(
                prepare_replay_source if prepared.get("paired_replay") else prepare_source,
                home,
                prepared["spec"],
                prepared["requested_cases"] if prepared.get("paired_replay") else list(pending.values()),
                prepared["fingerprints"],
                {**context, "base_commit": base, "source_validation": path},
            )
            if group is None:
                continue
        report = await _review_source(wrapper, home, group, memory_limit_mib=memory_limit_mib)
        if "equivalence_report" not in selection:
            selection["equivalence_report"] = report
        group_context = group["context"]
        path = group_context["source_validation"]
        selection["source_reviews"][path] = {
            **group_context,
            "case_ids": [case.case_id for case in group["cases"]],
            "equivalence_report": report,
        }
        if report["decision"] == "construction_unchanged":
            if not approved:
                selection.update(group_context, equivalence_report=report)
            for case in group["cases"]:
                approved[case.case_id] = {"result_dir": path, "commit_sha": group_context["base_commit"]}
                pending.pop(case.case_id)
            if prepared.get("paired_replay"):
                selection["replay_selection"] = group["replay_selection"]
                break
    if approved:
        targets = set(target_case_ids)
        selection["case_sources"] = {key: source for key, source in approved.items() if key in targets}
        if set(approved) - targets:
            draw = select_query_only()
            selection.update({key: value for key, value in draw.items() if key != "selected"})
            selection["probability_selected"] = draw["selected"]
            if draw["selected"]:
                selection["case_sources"].update(approved)
        selection["selected"] = bool(selection["case_sources"])
        selection["reason"] = (
            "target_cases_reused"
            if selection["case_sources"] and targets
            else (
                "query_only_selected_by_60_percent_policy"
                if selection["selected"]
                else "full_validation_selected_by_60_percent_policy"
            )
        )
    else:
        selection["reason"] = "construction_not_equivalent"
    selection["rebuild_reasons"] = {
        case.case_id: (
            "full_validation_selected_by_60_percent_policy"
            if case.case_id in approved and selection["probability_selected"] is False
            else "no_compatible_reviewed_snapshot"
        )
        for case in prepared.get("requested_cases", prepared["cases"])
        if case.case_id not in selection["case_sources"]
    }
    selection["reuse_reasons"] = {
        key: "target_case" if key in target_case_ids else "60_percent_policy" for key in selection["case_sources"]
    }
    return selection
