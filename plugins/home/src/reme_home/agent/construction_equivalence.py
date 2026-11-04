"""Construction equivalence review over isolated source and construction exports."""

import shutil
from typing import Literal

from pydantic import Field

from reme.utils.async_utils import complete_in_thread as blocking

from ..evaluation.artifacts import contained, file_sha256, inspect_archive, read_json, write_json
from ..evaluation.inputs import reuse_inputs
from ..preparation.code import tree_fingerprint
from ..preparation.spec import SpecModel
from .candidate_gate import prepare_review_scope, review_export
from .workflow_common import review_report


class ConstructionEquivalenceReport(SpecModel):
    """Only an explicit validated unchanged decision can authorize replay."""

    decision: Literal["construction_changed", "construction_unchanged"]
    reason: str = Field(min_length=1, pattern=r"\S")


def prepare_equivalence_scope(home, context, spec, cases, fingerprints, *, root):
    """Copy only pinned source, fixed runtime and verified pre-query construction evidence."""
    record = read_json(contained(home, "metadata/preparation.json"))
    runtime = contained(home, "code/runtime", directory=True)
    if tree_fingerprint(runtime) != record["code"]["runtime_fingerprint"]:
        raise ValueError("Fixed runtime has changed since preparation")
    _, archives = reuse_inputs(home, spec, cases, fingerprints)
    scope = prepare_review_scope(home, context, root=root)
    shutil.copytree(runtime, root / "runtime")
    write_json(root / "configuration.json", record["code"])
    source = contained(home, context["source_validation"], directory=True)
    for case_id, (archive, expected_sha) in archives.items():
        target = root / "construction" / case_id
        target.mkdir(parents=True)
        result = read_json(contained(archive.parent, "result.json"))
        write_json(target / "result.json", {key: result.get(key) for key in ("success", "status", "workspace_sha256")})
        log = contained(archive.parent, "build.log")
        if log.is_file():
            shutil.copyfile(log, target / "build.log")
        # Validate every member, even excluded Git entries. Never copy query
        # outputs, follow links, or trust the previously extracted workspace.
        extracted = target / "workspace"
        inspect_archive(archive, spec.max_artifact_bytes, extracted)
        if file_sha256(archive) != expected_sha:
            raise ValueError(f"Construction source changed during review export: {case_id}")
        for path in sorted(extracted.rglob(".git"), key=lambda value: len(value.parts), reverse=True):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    write_json(root / "source.json", {"result_dir": str(source), "case_ids": list(archives)})
    return scope


async def review_construction(wrapper, home, context, spec, cases, fingerprints, *, memory_limit_mib=None):
    """Stop on sandbox setup errors; review errors conservatively require rebuilding."""
    async with review_export(home, "construction-equivalence") as root:
        scope = await blocking(prepare_equivalence_scope, home, context, spec, cases, fingerprints, root=root)
        return await review_report(
            wrapper,
            home,
            scope,
            f"Compare construction from {context['base_commit']} to {context['candidate_commit']}.\n"
            f"Read-only source, diff and construction evidence: {scope['scratch']}",
            {**context, "source_path": scope["scratch"]},
            role="construction_equivalence",
            sandbox_label="Construction review",
            category="construction-equivalence",
            report_model=ConstructionEquivalenceReport,
            failure_decision="construction_changed",
            failure_reason="Equivalence review failed; rebuild construction",
            memory_limit_mib=memory_limit_mib,
        )
