"""Focused tests for Meta-ReMe training-data preparation."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import yaml

META_REME = Path(__file__).resolve().parents[2] / "meta-reme"
sys.path.insert(0, str(META_REME))
SPEC = importlib.util.spec_from_file_location("meta_reme_run", META_REME / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)

from data_preparation import beam  # noqa: E402  pylint: disable=wrong-import-position


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

    manifest = json.loads(workspace.path("datasets/search/manifest.json").read_text(encoding="utf-8"))
    case_files = list(workspace.path("datasets/search/cases").glob("*.json"))
    stored_case = json.loads(case_files[0].read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    assert manifest["query_count"] == 1
    assert stored_case["case_id"] == "train"
    stored_text = "".join(path.read_text(encoding="utf-8") for path in case_files)
    assert "held-out" not in stored_text
    assert not case_files[0].stat().st_mode & 0o222

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
    assert workspace.path("datasets/search/manifest.json").is_file()


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
            },
        ),
        encoding="utf-8",
    )

    config = run.load_config(config_path)

    assert config["meta_workspace"] == run.PROJECT_ROOT / "benchmark/beam/meta-workspace"
    assert config["dataset_source"] == run.PROJECT_ROOT / "benchmark/beam/dataset/BEAM"
    assert config["dataset_variant"] == "100K"
    assert config["train_case_ids"] == ["1", "2"]
