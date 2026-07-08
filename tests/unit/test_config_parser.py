"""Tests for configuration parsing helpers."""

from pathlib import Path

import pytest

from reme.config.config_parser import (
    _expand_env_vars,
    _load_config,
    _read_config_file,
    parse_args,
    parse_dot_notation,
)


def test_load_builtin_config_by_filename_with_suffix():
    """Built-in config names may include the YAML suffix."""
    cfg = _load_config("default.yaml")

    assert cfg["service"]["backend"] == "http"


def test_default_config_registers_daily_write_job():
    """``daily_write`` is exposed as a base job backed by ``daily_write_step``."""
    cfg = _load_config("default.yaml")

    job = cfg["jobs"]["daily_write"]
    assert job["backend"] == "base"
    assert job["steps"] == [{"backend": "daily_write_step"}]
    assert job["parameters"]["required"] == ["name", "description", "session_id", "content"]


def test_default_config_keeps_frontmatter_chunk_metadata_opt_in():
    """Markdown frontmatter-to-chunk metadata is disabled by default for compatibility."""
    cfg = _load_config("default.yaml")

    markdown = cfg["components"]["file_chunker"]["markdown"]
    assert markdown["include_frontmatter_in_metadata"] is False
    # Allow-list defaults to empty; combined with the False above, chunk metadata stays empty.
    assert markdown["include_frontmatter_keys_in_metadata"] == [] or markdown.get(
        "include_frontmatter_keys_in_metadata",
    ) in (None, [])


def test_parse_args_rejects_non_key_value_extra_argument():
    """Extra CLI arguments must use key=value syntax."""
    with pytest.raises(ValueError, match="expected key=value"):
        parse_args("search", "hello")


@pytest.mark.parametrize("item", ["=1", ".a=1", "a.=1", "a..b=1"])
def test_parse_dot_notation_rejects_empty_key_segments(item):
    """Dot notation keys cannot contain empty path segments."""
    with pytest.raises(ValueError, match="Invalid dot notation key"):
        parse_dot_notation([item])


def test_read_config_file_rejects_non_mapping_root(tmp_path: Path):
    """Config files must contain a mapping at the root."""
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config root must be a mapping"):
        _read_config_file(config_path)


def test_expand_env_vars_converts_expanded_scalar_types(monkeypatch):
    """Expanded environment values keep YAML scalar typing."""
    monkeypatch.setenv("PORT", "18080")
    monkeypatch.setenv("ENABLED", "false")

    expanded = _expand_env_vars(
        {
            "port": "${PORT}",
            "enabled": "${ENABLED}",
            "zip": "${ZIP:-007}",
            "url": "http://${HOST:-localhost}:${PORT}",
            "string_bool": '${STRING_BOOL:-"false"}',
        },
    )

    assert expanded == {
        "port": 18080,
        "enabled": False,
        "zip": "007",
        "url": "http://localhost:18080",
        "string_bool": "false",
    }


def test_jinli_lme_config_registers_lme_one_question_job():
    """``jinli_lme.yaml`` exposes a ``lme_one_question`` job that bundles the
    LME ``context_answer`` and ``answer_judge`` steps and requires every
    input the two steps consume."""
    cfg = _load_config("jinli_lme.yaml")

    job = cfg["jobs"]["lme_one_question"]
    assert job["backend"] == "base"
    assert [step["backend"] for step in job["steps"]] == [
        "context_answer_step",
        "answer_judge_step",
    ]
    assert job["parameters"]["required"] == [
        "query",
        "session_context",
        "current_date",
        "agent_answer",
        "golden_answer",
        "question_type",
    ]
    props = set(job["parameters"]["properties"])
    assert props == set(job["parameters"]["required"])
