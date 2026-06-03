"""Tests for PromptHandler."""

# pylint: disable=missing-function-docstring

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from reme4.components.prompt_handler import PromptHandler


# -- init & load_prompt_dict --------------------------------------------------


def test_init_filters_non_string_values():
    ph = PromptHandler(greeting="hello", count=42, items=[1, 2])
    assert ph.data == {"greeting": "hello"}


def test_load_prompt_dict_basic():
    ph = PromptHandler()
    ph.load_prompt_dict({"a": "alpha", "b": "beta"})
    assert ph.data == {"a": "alpha", "b": "beta"}


def test_load_prompt_dict_skips_non_string_values():
    ph = PromptHandler()
    ph.load_prompt_dict({"good": "ok", "bad": 123})
    assert ph.data == {"good": "ok"}


def test_load_prompt_dict_overwrite_true():
    ph = PromptHandler(a="old")
    ph.load_prompt_dict({"a": "new"}, overwrite=True)
    assert ph.data["a"] == "new"


def test_load_prompt_dict_overwrite_false():
    ph = PromptHandler(a="old")
    ph.load_prompt_dict({"a": "new"}, overwrite=False)
    assert ph.data["a"] == "old"


def test_load_prompt_dict_none():
    ph = PromptHandler(a="old")
    result = ph.load_prompt_dict(None)
    assert result is ph
    assert ph.data == {"a": "old"}


def test_load_prompt_dict_non_dict():
    ph = PromptHandler()
    result = ph.load_prompt_dict("not a dict")
    assert result is ph
    assert ph.data == {}


# -- load from file -----------------------------------------------------------


def test_load_prompt_by_file_yaml():
    data = {"greeting": "Hello {name}", "farewell": "Goodbye"}
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(data, f)
        f.flush()
        ph = PromptHandler()
        ph.load_prompt_by_file(f.name)
    assert ph.data["greeting"] == "Hello {name}"
    assert ph.data["farewell"] == "Goodbye"
    Path(f.name).unlink()


def test_load_prompt_by_file_json():
    data = {"q1": "What is {topic}?"}
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(data, f)
        f.flush()
        ph = PromptHandler()
        ph.load_prompt_by_file(f.name)
    assert ph.data["q1"] == "What is {topic}?"
    Path(f.name).unlink()


def test_load_prompt_by_file_none():
    ph = PromptHandler()
    result = ph.load_prompt_by_file(None)
    assert result is ph


def test_load_prompt_by_file_nonexistent():
    ph = PromptHandler()
    result = ph.load_prompt_by_file("/nonexistent/path.yaml")
    assert result is ph
    assert ph.data == {}


def test_load_prompt_by_file_unsupported_extension():
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("hello")
        f.flush()
        ph = PromptHandler()
        result = ph.load_prompt_by_file(f.name)
    assert result is ph
    assert ph.data == {}
    Path(f.name).unlink()


# -- get_prompt & i18n --------------------------------------------------------


def test_get_prompt_bare_key():
    ph = PromptHandler(greeting="Hello")
    assert ph.get_prompt("greeting") == "Hello"


def test_get_prompt_strips():
    ph = PromptHandler(greeting="  Hello  \n")
    assert ph.get_prompt("greeting") == "Hello"


def test_get_prompt_missing_raises():
    ph = PromptHandler()
    with pytest.raises(KeyError, match="not found"):
        ph.get_prompt("missing")


def test_get_prompt_language_fallback():
    ph = PromptHandler(language="zh", greeting="Hello", greeting_zh="你好")
    assert ph.get_prompt("greeting") == "你好"


def test_get_prompt_language_fallback_to_bare():
    ph = PromptHandler(language="zh", greeting="Hello")
    assert ph.get_prompt("greeting") == "Hello"


def test_has_prompt():
    ph = PromptHandler(greeting="Hello")
    assert ph.has_prompt("greeting") is True
    assert ph.has_prompt("missing") is False


def test_has_prompt_with_language():
    ph = PromptHandler(language="en", greeting_en="Hi")
    assert ph.has_prompt("greeting") is True


# -- list_prompts -------------------------------------------------------------


def test_list_prompts_all():
    ph = PromptHandler(a="1", b_en="2", c_zh="3")
    assert sorted(ph.list_prompts()) == ["a", "b_en", "c_zh"]


def test_list_prompts_filtered():
    ph = PromptHandler(a="1", b_en="2", c_en="3", d_zh="4")
    assert sorted(ph.list_prompts("en")) == ["b_en", "c_en"]


# -- prompt_format (flag filtering) -------------------------------------------


def test_flag_filter_keeps_matching():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "[verbose] extra detail\nalways here"})
    result = ph.prompt_format("p", verbose=True)
    assert "extra detail" in result
    assert "always here" in result


def test_flag_filter_removes_non_matching():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "[verbose] extra detail\nalways here"})
    result = ph.prompt_format("p", verbose=False)
    assert "extra detail" not in result
    assert "always here" in result


def test_flag_filter_default_false():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "[debug] debug info\nbase"})
    # When no flags are passed at all, _apply_flag_filter is not called,
    # so flagged lines are kept as-is (including the tag text after regex sub).
    result = ph.prompt_format("p", debug=False)
    assert "debug info" not in result
    assert "base" in result


def test_flag_filter_unflagged_lines_always_kept():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "line1\nline2\nline3"})
    result = ph.prompt_format("p")
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result


# -- prompt_format (variable substitution) ------------------------------------


def test_format_variables():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "Hello {name}, welcome to {place}"})
    result = ph.prompt_format("p", name="Alice", place="Wonderland")
    assert result == "Hello Alice, welcome to Wonderland"


def test_format_missing_variable_raises():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "Hello {name}"})
    with pytest.raises(ValueError, match="Missing format variables"):
        ph.prompt_format("p")


def test_format_missing_variable_no_validate():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "Hello {name}"})
    result = ph.prompt_format("p", validate=False)
    assert "{name}" in result


def test_format_no_variables_no_error():
    ph = PromptHandler()
    ph.load_prompt_dict({"p": "No vars here"})
    result = ph.prompt_format("p")
    assert result == "No vars here"


# -- prompt_format (combined flags + variables) -------------------------------


def test_format_flags_and_variables_combined():
    ph = PromptHandler()
    ph.load_prompt_dict(
        {
            "p": "[verbose] Debug: {detail}\nResult: {answer}",
        },
    )
    result = ph.prompt_format("p", verbose=True, detail="trace", answer="42")
    assert "Debug: trace" in result
    assert "Result: 42" in result


def test_format_flags_false_variable_not_needed():
    ph = PromptHandler()
    ph.load_prompt_dict(
        {
            "p": "[verbose] Debug: {detail}\nResult: {answer}",
        },
    )
    result = ph.prompt_format("p", verbose=False, answer="42")
    assert "Debug" not in result
    assert "Result: 42" in result


# -- repr ---------------------------------------------------------------------


def test_repr():
    ph = PromptHandler(language="en", a="1", b="2")
    r = repr(ph)
    assert "en" in r
    assert "2" in r


if __name__ == "__main__":
    print("\n=== PromptHandler Tests ===")
    test_init_filters_non_string_values()
    test_load_prompt_dict_basic()
    test_load_prompt_dict_skips_non_string_values()
    test_load_prompt_dict_overwrite_true()
    test_load_prompt_dict_overwrite_false()
    test_load_prompt_dict_none()
    test_load_prompt_dict_non_dict()
    test_load_prompt_by_file_yaml()
    test_load_prompt_by_file_json()
    test_load_prompt_by_file_none()
    test_load_prompt_by_file_nonexistent()
    test_load_prompt_by_file_unsupported_extension()
    test_get_prompt_bare_key()
    test_get_prompt_strips()
    test_get_prompt_missing_raises()
    test_get_prompt_language_fallback()
    test_get_prompt_language_fallback_to_bare()
    test_has_prompt()
    test_has_prompt_with_language()
    test_list_prompts_all()
    test_list_prompts_filtered()
    test_flag_filter_keeps_matching()
    test_flag_filter_removes_non_matching()
    test_flag_filter_default_false()
    test_flag_filter_unflagged_lines_always_kept()
    test_format_variables()
    test_format_missing_variable_raises()
    test_format_missing_variable_no_validate()
    test_format_no_variables_no_error()
    test_format_flags_and_variables_combined()
    test_format_flags_false_variable_not_needed()
    test_repr()
    print("\n所有测试通过!")
