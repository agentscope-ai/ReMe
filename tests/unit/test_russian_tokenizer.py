"""Tests for the Russian tokenizer."""

# pylint: disable=protected-access

import hashlib
import json

import pytest

from reme.components.keyword_index.bm25_index import BM25Index
from reme.components.tokenizer import RegexTokenizer, RussianTokenizer
from reme.core.registry import R
from reme.enumeration import ComponentEnum
from reme.schema import ComponentConfig

pytest.importorskip("snowballstemmer", reason="RussianTokenizer needs the snowball extras")

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

# The keys `main` fingerprints for a plain Regex tokenizer. The digest depends
# on the stopword file, so it is not asserted as a literal; what matters is that
# the key set stays exactly this and no implementation setting creeps in.
REGEX_CONFIG_KEYS = {"class", "filter_stopwords", "stopwords_sha256"}


def _index_over(tokenizer) -> BM25Index:
    """A BM25Index with just the tokenizer wired, without starting the index."""
    index = object.__new__(BM25Index)
    index.tokenizer = tokenizer
    return index


@pytest.mark.asyncio
async def test_case_forms_fold_to_one_token():
    """Words differing only by case ending must share one index term."""
    tok = RussianTokenizer(filter_stopwords=False)
    await tok.start()
    try:
        tokens = tok.tokenize(["инструментом инструмента инструменты"])[0]
    finally:
        await tok.close()

    assert len(tokens) == 3
    assert len(set(tokens)) == 1, f"case forms did not fold: {tokens}"
    assert tokens[0] == "инструмент"


@pytest.mark.asyncio
async def test_note_and_query_share_a_term():
    """The pair from the report must land on a common term."""
    tok = RussianTokenizer(filter_stopwords=False)
    await tok.start()
    try:
        note, query = tok.tokenize(["работаю инструментом каждый день", "инструмента"])
    finally:
        await tok.close()

    assert set(query) & set(note), f"{query} did not match {note}"


@pytest.mark.asyncio
async def test_non_cyrillic_tokens_are_untouched():
    """Latin and CJK tokens keep the plain regex behaviour."""
    tok = RussianTokenizer(filter_stopwords=False)
    regex = RegexTokenizer(filter_stopwords=False)
    await tok.start()
    await regex.start()
    try:
        text = "Python машина обучение 学习"
        ours = tok.tokenize([text])[0]
        base = regex.tokenize([text])[0]
    finally:
        await tok.close()
        await regex.close()

    for token in ("python", "学", "习"):
        assert token in ours, f"{token} missing from {ours}"

    ours_non_cyrillic = [t for t in ours if not set(t.lower()) & CYRILLIC]
    base_non_cyrillic = [t for t in base if not set(t.lower()) & CYRILLIC]
    assert ours_non_cyrillic == base_non_cyrillic


@pytest.mark.asyncio
async def test_non_cyrillic_tokens_keep_their_case_when_lower_is_false():
    """``lower=False`` must keep Latin casing; folding is not a lowercase pass.

    ``_postprocess()`` applies the caller's ``lower`` choice. Folding tokens
    beforehand lowercased Latin ones before that choice could take effect, so
    ``lower=False`` silently behaved like ``lower=True`` for them.
    """
    tok = RussianTokenizer(filter_stopwords=False)
    await tok.start()
    try:
        tokens = tok.tokenize(["Python OpenAI"], lower=False)[0]
    finally:
        await tok.close()

    assert tokens == ["Python", "OpenAI"], f"case was not preserved: {tokens}"


@pytest.mark.asyncio
async def test_pymorphy3_keeps_non_cyrillic_case_when_lower_is_false():
    """The same case-preservation rule must hold for the dictionary-form engine."""
    pytest.importorskip("pymorphy3", reason="RussianTokenizer needs the russian-full extras")
    tok = RussianTokenizer(engine="pymorphy3", filter_stopwords=False)
    await tok.start()
    try:
        tokens = tok.tokenize(["Python OpenAI"], lower=False)[0]
    finally:
        await tok.close()

    assert tokens == ["Python", "OpenAI"], f"case was not preserved: {tokens}"


@pytest.mark.asyncio
async def test_tokenize_before_start_falls_back_to_regex():
    """Before start() it must not crash and must behave like the regex tokenizer."""
    tok = RussianTokenizer(filter_stopwords=False)
    regex = RegexTokenizer(filter_stopwords=False)

    assert tok.tokenize(["инструментом"])[0] == regex.tokenize(["инструментом"])[0]


def test_unknown_engine_is_rejected():
    """An unknown engine name must fail fast instead of silently degrading."""
    with pytest.raises(ValueError, match="Unsupported Russian tokenizer engine"):
        RussianTokenizer(engine="nope")


def test_application_configuration_builds_the_tokenizer():
    """`backend=russian` must resolve through the registry and construct.

    `Application._instantiate()` looks the class up by `cfg.backend` and then
    forwards `cfg.model_dump()` to the constructor, so the registry key arrives
    as a keyword argument named `backend`. Treating that same argument as the
    folding engine made neither value work: `russian` raised as an unknown
    engine and the real engine names failed the registry lookup.
    """
    cfg = ComponentConfig(**{"backend": "russian", "filter_stopwords": False})
    tokenizer_cls = R.get(ComponentEnum.TOKENIZER, cfg.backend)
    assert tokenizer_cls is RussianTokenizer, f"registry returned {tokenizer_cls}"

    tok = tokenizer_cls(**cfg.model_dump())
    assert tok.backend == "russian"
    assert tok.engine == "snowball"


def test_application_configuration_can_select_the_engine():
    """The engine is a separate option, so it travels alongside the registry key."""
    cfg = ComponentConfig(**{"backend": "russian", "engine": "snowball", "filter_stopwords": False})
    tokenizer_cls = R.get(ComponentEnum.TOKENIZER, cfg.backend)

    tok = tokenizer_cls(**cfg.model_dump())
    assert tok.backend == "russian"
    assert tok.engine == "snowball"


@pytest.mark.asyncio
async def test_fingerprint_tracks_the_engine():
    """Switching the folding engine must change the index fingerprint."""
    tok = RussianTokenizer(filter_stopwords=False)
    await tok.start()
    try:
        index = _index_over(tok)
        assert index._tokenizer_config()["engine"] == "snowball"

        tok.engine = "pymorphy3"
        assert index._tokenizer_config()["engine"] == "pymorphy3"
    finally:
        await tok.close()


def test_application_created_regex_keeps_its_fingerprint():
    """A plain Regex tokenizer must not gain an engine entry.

    `Application._instantiate()` passes `backend="regex"` from the config, so an
    implementation that fingerprints any non-empty `backend` would rewrite the
    index of the default deployment even though its tokens never changed.
    """
    index = _index_over(RegexTokenizer(backend="regex", filter_stopwords=True))

    config = index._tokenizer_config()
    assert set(config) == REGEX_CONFIG_KEYS, f"the config gained a key: {config}"


def test_fingerprinting_the_registry_key_would_change_the_digest():
    """Why `backend` stays out: including it moves the digest of a Regex index.

    This is the failure the previous revision had. The assertion is deliberately
    about the digest moving rather than about a fixed value, because the stopword
    file is part of the payload.
    """

    def digest(payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]

    index = _index_over(RegexTokenizer(backend="regex", filter_stopwords=True))
    kept = index._tokenizer_config()

    polluted = dict(kept)
    polluted["backend"] = "regex"

    assert digest(kept) != digest(polluted), "the registry key must be distinguishable from an output setting"


def test_tokenizers_without_an_engine_keep_their_config():
    """A tokenizer that offers no implementation choice reports nothing."""
    index = _index_over(RegexTokenizer(filter_stopwords=False))

    assert "engine" not in index._tokenizer_config()


@pytest.mark.asyncio
async def test_custom_stopwords_survive_folding(tmp_path):
    """A stopword written in its dictionary form must still filter its folded form.

    Filtering runs in `_postprocess`, after the token was folded, so the set has
    to be folded as well. With `машины` in the file and `машин` in the token
    stream, an unfolded set lets the stopword through.
    """
    stopwords = tmp_path / "stopwords.txt"
    stopwords.write_text("машины\n", encoding="utf-8")

    tok = RussianTokenizer(stopwords_path=stopwords, filter_stopwords=True)
    await tok.start()
    try:
        tokens = tok.tokenize(["машины"])[0]
    finally:
        await tok.close()

    assert tokens == [], f"the stopword survived folding: {tokens}"


@pytest.mark.asyncio
async def test_pymorphy3_stopwords_survive_folding(tmp_path):
    """The same stopword case must hold for the dictionary-form engine."""
    pytest.importorskip("pymorphy3", reason="RussianTokenizer needs the russian-full extras")
    stopwords = tmp_path / "stopwords.txt"
    stopwords.write_text("машины\n", encoding="utf-8")

    tok = RussianTokenizer(engine="pymorphy3", stopwords_path=stopwords, filter_stopwords=True)
    await tok.start()
    try:
        tokens = tok.tokenize(["машины"])[0]
    finally:
        await tok.close()

    assert tokens == [], f"the stopword survived folding: {tokens}"
