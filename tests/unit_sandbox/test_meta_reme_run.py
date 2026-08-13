"""Focused tests for Meta-ReMe training-data preparation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
PROJECT_ROOT = META_REME.parent
for import_root in (META_REME, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
SPEC = importlib.util.spec_from_file_location("meta_reme_run", META_REME / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

from data_preparation import beam  # noqa: E402  pylint: disable=wrong-import-position


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _lme_item(case_id: str) -> dict:
    return {
        "question_id": case_id,
        "question_type": "single-session-user",
        "question": f"question {case_id}",
        "question_date": "2023/05/21 (Sun) 02:21",
        "answer": f"answer {case_id}",
        "original_answer": f"answer {case_id}",
        "answer_session_ids": [f"session-{case_id}"],
        "original_answer_session_ids": [f"session-{case_id}"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"],
        "haystack_session_ids": [f"session-{case_id}"],
        "haystack_sessions": [[{"role": "user", "content": f"memory {case_id}"}]],
    }


def test_prepare_workspace_stores_only_selected_longmemeval_cases(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps([_lme_item("train"), _lme_item("held-out")]), encoding="utf-8")
    workspace_root = tmp_path / "meta-workspace"

    workspace = run.prepare_workspace(
        workspace_root,
        "longmemeval",
        train_case_ids=["train"],
        dataset_source=source,
    )

    manifest = json.loads(workspace.path("dataset/manifest.json").read_text(encoding="utf-8"))
    case_files = list(workspace.path("dataset/cases").glob("*.json"))
    stored_case = json.loads(case_files[0].read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    assert manifest["query_count"] == 1
    assert stored_case["case_id"] == "train"
    stored_text = "".join(path.read_text(encoding="utf-8") for path in case_files)
    assert "held-out" not in stored_text
    assert not case_files[0].stat().st_mode & 0o222
    assert workspace.path("code/repo/reme/pyproject.toml").is_file()
    assert workspace.path("code/repo/reme/reme/config/lme.yaml").is_file()
    code_repository = workspace.path("code/repo/reme")
    assert _git(code_repository, "branch", "--show-current") == "init"
    assert _git(code_repository, "log", "-1", "--pretty=%s") == "Initial version"
    assert _git(code_repository, "status", "--short") == ""

    assert (
        run.prepare_workspace(
            workspace_root,
            "longmemeval",
            train_case_ids=["train"],
            dataset_source=source,
        ).root
        == workspace.root
    )


def test_prepare_workspace_initializes_an_existing_empty_directory(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps([_lme_item("train")]), encoding="utf-8")
    workspace_root = tmp_path / "meta-workspace"
    workspace_root.mkdir()

    workspace = run.prepare_workspace(workspace_root, "longmemeval", dataset_source=source)

    assert workspace.path(".meta-reme-workspace.json").is_file()
    assert workspace.path("dataset/manifest.json").is_file()
    assert workspace.path("code/repo/reme/reme/steps/benchmark/lme").is_dir()


def test_prepare_workspace_does_not_overwrite_existing_initial_code(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps([_lme_item("train")]), encoding="utf-8")
    workspace = run.prepare_workspace(tmp_path / "meta-workspace", "longmemeval", dataset_source=source)
    initial_head = _git(workspace.path("code/repo/reme"), "rev-parse", "HEAD")
    marker = workspace.path("code/repo/reme/user-change.txt")
    marker.write_text("keep me", encoding="utf-8")

    reopened = run.prepare_workspace(workspace.root, "longmemeval", dataset_source=source)

    assert reopened.root == workspace.root
    assert _git(workspace.path("code/repo/reme"), "rev-parse", "HEAD") == initial_head
    assert marker.read_text(encoding="utf-8") == "keep me"


def test_load_beam_cases_respects_variant_and_selection(tmp_path: Path) -> None:
    case_root = tmp_path / "chats/100K/7"
    (case_root / "probing_questions").mkdir(parents=True)
    (case_root / "chat.json").write_text(
        json.dumps(
            [
                {
                    "batch_number": 1,
                    "turns": [
                        [
                            {"role": "user", "content": "remember", "time_anchor": "March-15-2024"},
                            {"role": "assistant", "content": "okay"},
                        ],
                    ],
                },
            ],
        ),
        encoding="utf-8",
    )
    (case_root / "probing_questions/probing_questions.json").write_text(
        json.dumps({"information_extraction": [{"question": "what?", "ideal_answer": "remember"}]}),
        encoding="utf-8",
    )

    cases = beam.load_cases(tmp_path, "100K", ["7"])

    assert [case.case_id for case in cases] == ["7"]
    assert cases[0].metadata["variant"] == "100K"
    assert cases[0].queries[0].query_id == "information_extraction:1"
    assert cases[0].sessions[0].messages[1]["created_at"] == "2024-03-15T00:00:00"


def test_load_config_reads_all_preparation_arguments(tmp_path: Path) -> None:
    config_path = tmp_path / "config_meta_reme.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "meta_workspace": "benchmark/beam/meta-workspace",
                "dataset": {
                    "name": "beam",
                    "source": "benchmark/beam/dataset/BEAM",
                    "variant": "100K",
                    "train_case_ids": [1, "2"],
                },
                "validation": {"concurrency": 5, "fail_fast": True},
                "search": {
                    "max_agent_iters": 40,
                    "max_code_iterations": 3,
                },
            },
        ),
        encoding="utf-8",
    )

    config = run.load_config(config_path)

    assert config["meta_workspace"] == run.PROJECT_ROOT / "benchmark/beam/meta-workspace"
    assert config["dataset_source"] == run.PROJECT_ROOT / "benchmark/beam/dataset/BEAM"
    assert config["dataset_variant"] == "100K"
    assert config["train_case_ids"] == ["1", "2"]
    assert config["validation_concurrency"] == 5
    assert config["validation_fail_fast"] is True
    assert config["search"].max_agent_iters == 40
    assert config["search"].max_code_iterations == 3


def test_search_config_does_not_allow_model_overrides() -> None:
    with pytest.raises(ValueError, match=r"Unknown search configuration fields: \['fallback_model'\]"):
        run._load_search_config({"fallback_model": "qwen3.7-max"})

    with pytest.raises(ValueError, match=r"Unknown search configuration fields: \['model'\]"):
        run._load_search_config({"model": "another-model"})

    with pytest.raises(ValueError, match=r"Unknown search configuration fields: \['compression'\]"):
        run._load_search_config({"compression": True})


def test_search_config_allows_any_positive_code_iteration_limit() -> None:
    assert run._load_search_config({}).max_code_iterations == 5
    assert run._load_search_config({"max_code_iterations": 15}).max_code_iterations == 15


def test_prepare_and_validate_workspace_uses_all_installed_cases(tmp_path: Path) -> None:
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps([_lme_item("case-2"), _lme_item("case-1")]), encoding="utf-8")
    workspace_root = tmp_path / "meta-workspace"
    config_path = tmp_path / "config_meta_reme.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "meta_workspace": str(workspace_root),
                "dataset": {
                    "name": "longmemeval",
                    "source": str(source),
                    "train_case_ids": [],
                },
                "validation": {"concurrency": 5, "fail_fast": True},
            },
        ),
        encoding="utf-8",
    )
    calls = []

    code_repository = run.prepare_workspace(workspace_root, "longmemeval", dataset_source=source).path("code/repo/reme")
    commit = _git(code_repository, "rev-parse", "HEAD")

    def fake_validation(workspace, case_ids, concurrency, *, validation_id, fail_fast):
        calls.append((Path(workspace), case_ids, concurrency, validation_id, fail_fast))
        output = Path(workspace) / f"evaluations/init/{commit[:7]}/{validation_id}"
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(json.dumps({"commit_sha": commit}), encoding="utf-8")
        (output / "summary.json").write_text("{}\n", encoding="utf-8")
        return output

    workspace, validation = run.prepare_and_validate_workspace(config_path, validation_runner=fake_validation)

    assert validation == workspace.path(f"evaluations/init/{commit[:7]}/initial")
    assert run.TOOL_RUNTIME.workspace == workspace.root
    assert run.TOOL_RUNTIME.validation_concurrency == 5
    assert calls == [(workspace.root, ["case-2", "case-1"], 5, "initial", True)]

    _, repeated_validation = run.prepare_and_validate_workspace(config_path, validation_runner=fake_validation)

    assert repeated_validation == validation
    assert len(calls) == 1


def test_optimizer_starts_only_after_complete_search_validation(tmp_path: Path, monkeypatch) -> None:
    workspace_root = tmp_path / "workspace"
    source = tmp_path / "longmemeval.json"
    source.write_text(json.dumps([_lme_item("case-1"), _lme_item("case-2")]), encoding="utf-8")
    workspace = run.prepare_workspace(workspace_root, "longmemeval", dataset_source=source)
    commit = _git(workspace.path("code/repo/reme"), "rev-parse", "HEAD")
    validation = workspace.path(f"evaluations/init/{commit[:7]}/initial")
    validation.mkdir(parents=True)
    (validation / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "cases": [
                    {"case_id": "case-1", "status": "completed"},
                    {"case_id": "case-2", "status": "completed"},
                ],
            },
        ),
        encoding="utf-8",
    )
    calls = []

    monkeypatch.setattr(run, "create_search_model", lambda name: f"model:{name}")

    def factory(model, workspace_arg, **kwargs):
        calls.append(("factory", model, workspace_arg, kwargs))
        return "agent"

    async def runner(agent, *, objective):
        calls.append(("runner", agent, objective))
        return "result"

    result = asyncio.run(
        run.optimize_validated_workspace(
            workspace,
            validation,
            run.SearchConfig(max_code_iterations=3),
            validation_concurrency=2,
            optimizer_factory=factory,
            optimizer_runner=runner,
        ),
    )

    assert result == "result"
    assert calls[0][1] == "model:qwen3.8-max"
    assert calls[0][2] == workspace.root
    assert calls[0][3]["diagnostic_model"] == "model:qwen3.8-max"
    assert "at most 3 committed code iterations" in calls[1][2]

    (validation / "summary.json").write_text(
        json.dumps({"status": "completed", "cases": [{"case_id": "case-1"}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="complete every installed search case"):
        asyncio.run(
            run.optimize_validated_workspace(
                workspace,
                validation,
                run.SearchConfig(),
                validation_concurrency=2,
                optimizer_factory=factory,
                optimizer_runner=runner,
            ),
        )
