"""Host-owned ancestor selection and 60/40 construction replay policy."""

from pathlib import Path
import random
import subprocess
import tarfile

from ..code.repository import current_branch, require_clean, resolve_revision, validate_repository, validate_tree
from ..code.snapshot import tree
from ..preparation.code import git
from ..preparation.data.models import fingerprint
from .artifacts import contained, read_json
from .inputs import load_inputs, replay_sampling_eligibility, reuse_inputs
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


def reuse_context(home, spec, environment):
    """Find the nearest valid ancestor before review or sampling, under the workspace lock."""
    repo, record = validate_repository(home)
    require_clean(repo)
    commit = resolve_revision(repo, "HEAD")
    validate_tree(repo, record, commit)
    identity = {"candidate_commit": commit, "branch_name": current_branch(repo)}
    previous = previous_candidate_result(home, commit)
    if previous is not None:
        saved = read_json(contained(home, previous / "manifest.json")).get("automatic_construction_reuse")
        # A managed branch may move to the same failed commit. Retain its
        # commit-pinned review/draw while preserving the existing retry contract.
        return {"recovery": {**saved, **identity}} if saved is not None else None
    if commit == record["code"]["initial_commit"]:
        return None
    _, dataset, cases = load_inputs(home, spec)
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
    # Validate expensive archive checksums only in ancestry order, stopping at
    # the nearest usable source instead of re-reading every historical archive.
    for _, source, base in sorted(candidates):
        try:
            source_spec = spec.model_copy(update={"reuse_construction_from": source})
            eligible = replay_sampling_eligibility(home, source_spec, cases)
            reusable = [case for case in cases if case.case_id in eligible]
            if not reusable or (not spec.sampling and len(reusable) != len(cases)):
                continue
            reuse_inputs(home, source_spec, reusable, fingerprints)
            return {
                "context": {**identity, "base_commit": base, "source_validation": source},
                "spec": source_spec,
                "cases": reusable,
                "fingerprints": fingerprints,
            }
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError):
            continue
    return {**identity, "selected": False, "reason": "no_reusable_ancestor", "policy": "automatic_60_percent"}


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


async def automatic_reuse(wrapper, home, prepared, *, memory_limit_mib=None):
    """Automatic failures rebuild; only a sandboxed unchanged review permits a probability draw."""
    from ..agent.construction_equivalence import review_construction
    from ..agent.workflow_common import save_record
    from .component import blocking

    if prepared is None:
        return None
    if "recovery" in prepared:
        return prepared["recovery"]
    if "context" not in prepared:
        return prepared
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
    selection = {**context, "policy": "automatic_60_percent", "equivalence_report": report, "selected": False}
    if report["decision"] == "construction_unchanged":
        selection.update(select_query_only())
        selection["reason"] = (
            "query_only_selected_by_60_percent_policy"
            if selection["selected"]
            else ("full_validation_selected_by_60_percent_policy")
        )
    else:
        selection["reason"] = "construction_not_equivalent"
    return selection
