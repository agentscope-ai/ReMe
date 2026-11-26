"""Qwen tokenizer utility tests without external downloads."""

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reme.utils.qwen_token_tools import QwenTokenTools, estimate_token_count
from reme.utils import qwen_token_tools


class _Tokenizer:
    def encode(self, text, add_special_tokens):
        """Return one fake token per character."""
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(text), offsets=[(index, index + 1) for index in range(len(text))])


def test_custom_tokenizer_without_optional_dependency(monkeypatch):
    """Custom counting and truncation do not require the optional package."""
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    tools = QwenTokenTools(tokenizer=_Tokenizer())

    assert tools.count("你好 world") == 8
    assert tools.truncate("你好 world", 3) == ("你好 ", 3, True)
    assert tools.truncate("hi", 3) == ("hi", 2, False)


@pytest.fixture(name="word_tokenizer")
def _word_tokenizer():
    """Build a small tokenizer for loading and isolation regressions."""
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "one": 2, "two": 3, "three": 4, "four": 5},
            unk_token="[UNK]",
        ),
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return tokenizer


@pytest.mark.parametrize("source_kind", ["native", "wrapped", "file", "directory", "remote"])
def test_tokenizer_sources_disable_limits_and_isolate_instances(word_tokenizer, tmp_path, monkeypatch, source_kind):
    """Every source removes saved limits and isolates caller and instance mutations."""
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = word_tokenizer
    if source_kind == "wrapped":
        tokenizer = tokenizers.implementations.BaseTokenizer(tokenizer)
    tokenizer.enable_truncation(max_length=1)
    tokenizer.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
    original = tokenizer.to_str()
    if source_kind in ("native", "wrapped"):
        kwargs = {"tokenizer": tokenizer}
    else:
        path = tmp_path / "tokenizer.json"
        tokenizer.save(str(path))
        source = str(path if source_kind == "file" else tmp_path)
        if source_kind == "remote":
            source = f"test/{tmp_path.name}"
            load = Mock(return_value=tokenizer)
            monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", load)
        kwargs = {"tokenizer_source": source}
    first, second = QwenTokenTools(**kwargs), QwenTokenTools(**kwargs)
    text = "one two three four"
    for tools in (first, second):
        assert tools.count(text) == 4
        assert tools.count("") == 0
        assert tools.truncate(text, 1) == ("one", 1, True)
        assert tools.tokenizer is not tokenizer
    assert first.tokenizer is not second.tokenizer
    assert tokenizer.to_str() == original

    tokenizer.enable_truncation(max_length=2)
    tokenizer.enable_padding(length=10, pad_id=1, pad_token="[PAD]")
    first.tokenizer.enable_truncation(max_length=1)
    first.tokenizer.enable_padding(length=8, pad_id=1, pad_token="[PAD]")
    assert tokenizer.truncation["max_length"] == 2
    assert tokenizer.padding["length"] == 10
    assert second.count(text) == 4
    assert second.count("") == 0
    assert second.truncate(text, 1) == ("one", 1, True)
    if source_kind == "remote":
        load.assert_called_once_with(source)


@pytest.mark.parametrize(
    "failure, reason",
    [
        ("missing_dependency", "missing 'tokenizers' package"),
        ("invalid_json", "load failed"),
        ("remote", "load failed"),
    ],
)
def test_unavailable_tokenizer_warns_once_and_estimates(tmp_path, monkeypatch, failure, reason):
    """Real import, local JSON, and remote failures warn once and fall back."""
    source = f"test/{tmp_path.name}"
    if failure == "missing_dependency":
        monkeypatch.setitem(sys.modules, "tokenizers", None)
    else:
        tokenizers = pytest.importorskip("tokenizers")
        if failure == "invalid_json":
            path = tmp_path / "tokenizer.json"
            path.write_text("invalid JSON", encoding="utf-8")
            source = str(path)
        else:
            load = Mock(side_effect=OSError("offline"))
            monkeypatch.setattr(tokenizers.Tokenizer, "from_pretrained", load)
    tools = QwenTokenTools(source)

    with pytest.warns(RuntimeWarning, match=reason) as caught:
        assert tools.count("你好 world") == estimate_token_count("你好 world")
        assert tools.count("a") == 1
        assert tools.count("") == 0
        assert tools.truncate("abcdefghij", 1) == ("abcde", 1, True)
        assert tools.truncate("你好", 1) == ("你", 1, True)
        assert tools.truncate("abc", 1) == ("abc", 1, False)
        assert tools.truncate("", 1) == ("", 0, False)
        assert tools.tokenizer is None
    assert len(caught) == 1
    assert "pip install 'reme-ai[qwen-tokenizer]'" in str(caught[0].message)
    if failure == "remote":
        load.assert_called_once_with(source)


@pytest.mark.parametrize("failure", [None, RuntimeError])
def test_concurrent_initialization_loads_and_warns_once(monkeypatch, failure):
    """Concurrent property and count access share one initialization attempt."""
    load_attempts = []
    tokenizer = _Tokenizer()
    ready = Barrier(8)

    def load(source):
        load_attempts.append(source)
        sleep(0.01)  # Allow competing callers to reach the initialization lock.
        if failure is not None:
            raise failure("unavailable")
        return tokenizer

    def access(index):
        ready.wait(timeout=5)
        return tools.tokenizer if index % 2 else tools.count("hello")

    monkeypatch.setattr(qwen_token_tools, "_load_tokenizer", load)
    tools = QwenTokenTools()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(access, range(8)))
        assert tools.tokenizer is (None if failure else tokenizer)
    assert load_attempts == [tools.tokenizer_source]
    assert results[::2] == [1 if failure else 5] * 4
    assert all(result is (None if failure else tokenizer) for result in results[1::2])
    assert len(caught) == (1 if failure else 0)


@pytest.fixture(name="byte_tokenizer")
def _byte_tokenizer():
    """Build a byte-level tokenizer locally, including partial Unicode tokens."""
    tokenizers = pytest.importorskip("tokenizers")
    alphabet = tokenizers.pre_tokenizers.ByteLevel.alphabet()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE({char: index for index, char in enumerate(alphabet)}, []))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    return tokenizer


@pytest.mark.parametrize("text", ["🦄abc", "𠮷abc", "e\u0301abc"])
def test_unicode_truncation_preserves_original_prefix(byte_tokenizer, text):
    """Never decode incomplete UTF-8 tokens or change the source spelling."""
    tools = QwenTokenTools(tokenizer=byte_tokenizer)
    for budget in range(1, tools.count(text) + 1):
        truncated, count, changed = tools.truncate(text, budget)
        assert text.startswith(truncated)
        assert "�" not in truncated
        assert count == tools.count(truncated)
        assert count <= budget
        assert changed == (truncated != text)
    if text[0] != "e":
        assert tools.truncate(text, 1) == ("", 0, True)
        assert tools.truncate(text, 4) == (text[0], 4, True)


def test_normalized_truncation_preserves_original_prefix():
    """NFC output must never replace the original decomposed spelling."""
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordPiece({"[UNK]": 0, "é": 1, "##a": 2, "##b": 3, "##c": 4}),
    )
    tokenizer.normalizer = tokenizers.normalizers.NFC()
    tools = QwenTokenTools(tokenizer=tokenizer)
    text = "e\u0301abc"
    assert tools.count(text) == 4
    truncated, count, changed = tools.truncate(text, 1)
    assert truncated
    assert text.startswith(truncated)
    assert count == tools.count(truncated) == 1
    assert changed is True


def test_special_tokens_are_counted_without_automatic_wrapping(byte_tokenizer):
    """Explicit special tokens count, while the post-processor adds nothing."""
    tokenizers = pytest.importorskip("tokenizers")
    byte_tokenizer.add_special_tokens([tokenizers.AddedToken("[SPECIAL]", lstrip=True, rstrip=True)])
    byte_tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[SPECIAL] $A [SPECIAL]",
        special_tokens=[("[SPECIAL]", 256)],
    )
    tools = QwenTokenTools(tokenizer=byte_tokenizer)
    assert tools.count("") == 0
    assert tools.count("[SPECIAL]abc") == 4
    assert tools.truncate(" [SPECIAL] abc", 1) == (" [SPECIAL] ", 1, True)


@pytest.mark.parametrize("source", ["tokenizer.json", "."])
def test_local_sources_follow_file_updates_and_current_directory(word_tokenizer, tmp_path, monkeypatch, source):
    """New instances read updated files and resolve relative paths in the current directory."""
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.chdir(tmp_path)
    word_tokenizer.save("tokenizer.json")
    first = QwenTokenTools(source)
    assert first.count("one two") == 2

    word_tokenizer.normalizer = tokenizers.normalizers.Replace(" ", "")
    word_tokenizer.save("tokenizer.json")
    assert QwenTokenTools(source).count("one two") == 1
    assert first.count("one two") == 2

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    word_tokenizer.normalizer = None
    word_tokenizer.save("tokenizer.json")
    assert QwenTokenTools(source).count("one two") == 2


def test_invalid_inputs_do_not_load_tokenizer(monkeypatch):
    """Handle empty text and reject invalid inputs before loading or downloading."""
    load = Mock(side_effect=AssertionError("unexpected tokenizer load"))
    monkeypatch.setattr(qwen_token_tools, "_load_tokenizer", load)
    tools = QwenTokenTools()
    assert tools.count("") == 0
    assert tools.truncate("", 1) == ("", 0, False)
    for text in (None, b"hello"):
        with pytest.raises(TypeError, match="text must be a string"):
            tools.count(text)
        with pytest.raises(TypeError, match="text must be a string"):
            tools.truncate(text, 1)
    for budget in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            tools.truncate("hello", budget)
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            tools.truncate("", budget)
    load.assert_not_called()
