"""Construction equivalence review over isolated source and construction exports."""

import json
from pathlib import Path
import shutil
import sys
from typing import Literal

from agentscope.tool import Glob, Grep, Read, Toolkit
from pydantic import Field
import yaml

from ..auto_bash import AutoBash
from ..evaluation.artifacts import contained, file_sha256, inspect_archive, read_json, write_json
from ..evaluation.component import blocking
from ..evaluation.inputs import reuse_inputs
from ..preparation.code import tree_fingerprint
from ..preparation.spec import SpecModel
from .candidate_gate import prepare_review_scope, review_export
from .workflow_common import WorkflowSandbox, save_record


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
        return await _review_construction(wrapper, home, context, scope, memory_limit_mib)


async def _review_construction(wrapper, home, context, scope, memory_limit_mib):
    """Run inside the caller-owned export lifetime; persist only the report."""
    backend = WorkflowSandbox(scope, memory_limit_mib)
    probe = await backend.exec_shell(
        [sys.executable, "-I", "-B", "-c", "print('construction review sandbox')"],
        timeout=10,
    )
    if not probe.ok():
        raise RuntimeError("Construction review sandbox unavailable: " + probe.stderr.decode(errors="replace"))
    tools = [
        Read(backend=backend),
        Glob(backend=backend),
        Grep(backend=backend),
        AutoBash(cwd=scope["scratch"], backend=backend, is_read_only=True),
    ]
    for tool in tools:
        tool.is_concurrency_safe = False
    prompt = yaml.safe_load(Path(__file__).with_suffix(".yaml").read_text(encoding="utf-8"))
    report = {**context, "source_path": scope["scratch"]}
    try:
        reply = await wrapper.reply(
            f"Compare construction from {context['base_commit']} to {context['candidate_commit']}.\n"
            f"Read-only source, diff and construction evidence: {scope['scratch']}",
            system_prompt=prompt["system_prompt"],
            toolkit=Toolkit(tools=tools),
            builtin_tools=[],
            job_tools=[],
            skills=[],
            resume=None,
            session_id=None,
            fork_session=False,
            output_schema=None,
            _structured_report_model=ConstructionEquivalenceReport,
            permission_mode="bypass",
        )
        report.update({key: reply.get(key) for key in ("session_id", "usage", "result")})
        text = reply["result"].strip()
        if text.startswith("```") and text.endswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        report.update(ConstructionEquivalenceReport.model_validate(json.loads(text)).model_dump())
    except Exception as exc:
        report.update(
            decision="construction_changed",
            reason=f"Equivalence review failed; rebuild construction: {type(exc).__name__}: {exc}",
            review_failed=True,
        )
    path = await blocking(save_record, home, "construction-equivalence", report)
    return {**report, "review_path": path}
