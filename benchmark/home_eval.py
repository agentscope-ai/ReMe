"""Standalone test evaluation shared by the BEAM and LongMemEval entry points."""

import argparse
import asyncio
import json
from pathlib import Path

from pydantic import Field, model_validator
import yaml

from reme_home.code.repository import resolve_revision, validate_repository
from reme_home.evaluation.artifacts import contained, read_json
from reme_home.evaluation.component import blocking
from reme_home.evaluation.inputs import EvaluationSpec, load_inputs
from reme_home.judge import JUDGE_PLUGINS
from reme_home.preparation.code import tree_fingerprint
from reme_home.preparation.data.models import fingerprint
from reme_home.preparation.evaluation_source import EvaluationSource
from reme_home.preparation.spec import BenchmarkSpec, DebugSpec, SpecModel

from reme.application import Application

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestRunConfig(SpecModel):
    """Explicit dataset selection and execution options for one pinned candidate."""

    source_workspace: str = Field(min_length=1)
    test_workspace: str = Field(min_length=1)
    code_commit: str = Field(min_length=4)
    benchmark: BenchmarkSpec
    exclude_training_cases: bool = False
    evaluation: EvaluationSpec = Field(default_factory=lambda: EvaluationSpec(record_history=False))

    @model_validator(mode="after")
    def validate_test_mode(self):
        """Test runs score every prepared query and never consume optimizer budgets."""
        options = self.evaluation
        if any(
            (
                options.debug,
                options.sampling,
                options.record_history,
                options.max_evaluations is not None,
                options.sampling_policy is not None,
                options.case_ids is not None,
                options.query_ids is not None,
                options.reuse_construction_from is not None,
            ),
        ):
            raise ValueError(
                "Test evaluation requires full scoring with record_history=false; select cases in benchmark",
            )
        if self.exclude_training_cases and (self.benchmark.name != "longmemeval" or not self.benchmark.all_cases):
            raise ValueError("exclude_training_cases requires LongMemEval with all_cases=true")
        return self


def project_path(value: str, project_root: Path) -> Path:
    """Resolve runner paths independently of the launching working directory."""
    if value.startswith("~"):
        raise ValueError("Use absolute or project-relative paths, not home-relative paths")
    return (project_root / value).resolve()


def load_request(config_path, *, commit=None, source_workspace=None, test_workspace=None, project_root=PROJECT_ROOT):
    """Validate all source inputs before starting an application or writing test data."""
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    for key, value in (
        ("code_commit", commit),
        ("source_workspace", source_workspace),
        ("test_workspace", test_workspace),
    ):
        if value is not None:
            raw[key] = value
    # Execution settings can be partial while retaining the test-only history default.
    raw["evaluation"] = {"record_history": False, **raw.get("evaluation", {})}
    config = TestRunConfig.model_validate(raw)
    source_path = project_path(config.source_workspace, project_root)
    target = project_path(config.test_workspace, project_root)
    if target.is_relative_to(source_path) or source_path.is_relative_to(target):
        raise ValueError("Test and source workspaces must be separate, non-nested directories")
    if target.exists() and (
        not target.is_dir() or (any(target.iterdir()) and not (target / "home/metadata/workspace.json").is_file())
    ):
        raise ValueError("Use an empty test workspace or a compatible prepared Home test workspace")
    source = EvaluationSource.load(source_path, config.code_commit)
    benchmark = config.benchmark.model_copy(deep=True)
    if benchmark.name != source.spec.benchmark.name:
        raise ValueError("Test benchmark must match the source experiment")
    benchmark.source = str(project_path(benchmark.source, project_root).resolve(strict=True))
    if benchmark.evidence_source is not None:
        benchmark.evidence_source = str(project_path(benchmark.evidence_source, project_root).resolve(strict=True))
    if config.exclude_training_cases:
        items = json.loads(Path(benchmark.source).read_text(encoding="utf-8"))
        ids = [str(item["question_id"]) for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate LongMemEval question IDs")
        training = set(source.record["dataset"]["case_ids"])
        if training - set(ids):
            raise ValueError("Test dataset is missing training IDs; provide the original full dataset")
        benchmark.case_ids = [case_id for case_id in ids if case_id not in training]
        benchmark.all_cases = False
        benchmark = BenchmarkSpec.model_validate(benchmark.model_dump())
    spec = source.spec.model_copy(deep=True)
    spec.benchmark = benchmark
    spec.debug = DebugSpec(enabled=False)
    return config, target, source, spec


def existing_preparation(home, source, spec):
    """Reopen only the same request with unchanged code, runtime and normalized data."""
    marker = contained(home, "metadata/preparation.json")
    if not marker.exists():
        return None
    record = read_json(marker)
    expected = {**source.provenance, "request_fingerprint": fingerprint(spec.model_dump(mode="json"))}
    if record.get("test_evaluation") != expected:
        raise ValueError("Test workspace belongs to a different candidate or dataset; use a new workspace")
    domain = yaml.safe_load(contained(home, "domain_spec.yaml").read_text(encoding="utf-8"))
    if fingerprint(domain) != record["config_fingerprint"]:
        raise ValueError("Test preparation configuration has changed")
    repository, _ = validate_repository(home, record)
    if (
        resolve_revision(repository, "HEAD") != record["code"]["initial_commit"]
        or tree_fingerprint(repository) != record["code"]["candidate_fingerprint"]
        or tree_fingerprint(contained(home, "code/runtime", directory=True)) != source.provenance["runtime_fingerprint"]
    ):
        raise ValueError("Prepared test code or runtime has changed")
    load_inputs(home, EvaluationSpec(record_history=False))
    return record


async def run(
    config_path,
    *,
    commit=None,
    source_workspace=None,
    test_workspace=None,
    prepare_only=False,
    expected_benchmark=None,
):
    """Prepare transactionally, then score through the normal application lifecycle."""
    config, target, source, spec = await blocking(
        load_request,
        config_path,
        commit=commit,
        source_workspace=source_workspace,
        test_workspace=test_workspace,
    )
    if expected_benchmark is not None and config.benchmark.name != expected_benchmark:
        raise ValueError(f"This entry point requires benchmark.name={expected_benchmark}")
    app = Application(
        workspace_dir=str(target),
        plugins=["home", JUDGE_PLUGINS[config.benchmark.name]],
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
    )
    try:
        await app.start()
        home = target / "home"
        prepared = await blocking(existing_preparation, home, source, spec)
        if prepared is None:
            preparation = app.context.components["home.preparation"]["default"]
            prepared = await preparation.prepare(spec=spec.model_dump(mode="json"), code_source=source)
        result = {
            "test_workspace": str(target),
            **source.provenance,
            "evaluation_commit_sha": prepared["code"]["initial_commit"],
            "case_count": prepared["dataset"]["case_count"],
            "query_count": prepared["dataset"]["query_count"],
        }
        if not prepare_only:
            response = await app.run_job(
                "home_evaluate",
                **config.evaluation.model_dump(mode="json", exclude_unset=True),
            )
            if not response.success:
                raise RuntimeError(response.answer)
            result["evaluation"] = response.answer
        return result
    finally:
        await app.close()


def main(benchmark, default_config):
    """Explicit local one-shot entry point, with preparation-only service setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--commit", help="Candidate commit SHA or unambiguous hexadecimal prefix")
    parser.add_argument("--source-workspace", help="Prepared optimization workspace")
    parser.add_argument("--test-workspace", help="Independent test output workspace")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare for a later home_evaluate service call")
    args = parser.parse_args()
    result = asyncio.run(
        run(
            args.config,
            commit=args.commit,
            source_workspace=args.source_workspace,
            test_workspace=args.test_workspace,
            prepare_only=args.prepare_only,
            expected_benchmark=benchmark,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
