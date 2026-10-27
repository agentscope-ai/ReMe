"""Tests for Tokenizers."""

import asyncio

import pytest

from reme.components.tokenizer import JiebaTokenizer, RegexTokenizer


async def compare_tokenizers(texts: list[str], filter_stopwords: bool = False):
    """Compare both tokenizers on same input."""
    jieba = JiebaTokenizer(filter_stopwords=filter_stopwords)
    regex = RegexTokenizer(filter_stopwords=filter_stopwords)

    await jieba.start()
    await regex.start()

    jieba_result = jieba.tokenize(texts)
    regex_result = regex.tokenize(texts)

    await jieba.close()
    await regex.close()

    return jieba_result, regex_result


@pytest.mark.asyncio
async def test_mixed_chinese_english():
    """Test mixed Chinese-English text."""

    jieba_result, regex_result = await compare_tokenizers(
        ["我用 Python 学习 machine learning 和 iPhone15 Pro。"],
    )

    assert "python" in jieba_result[0]
    assert "python" in regex_result[0]


@pytest.mark.asyncio
async def test_open_example():
    """Test the 'open' example."""

    jieba_result, regex_result = await compare_tokenizers(
        ["我觉得open很好呀，能分好次吗？"],
    )

    # open 保持完整
    assert "open" in jieba_result[0]
    assert "open" in regex_result[0]

    # Regex 中文按字拆分
    assert "我" in regex_result[0]
    assert "很" in regex_result[0]


@pytest.mark.asyncio
async def test_with_stopwords():
    """Test with stopwords filtering."""

    jieba_result, regex_result = await compare_tokenizers(
        ["我觉得open很好呀，能分好次吗？"],
        filter_stopwords=True,
    )

    # 停用词被过滤
    assert "吗" not in jieba_result[0]
    assert "吗" not in regex_result[0]
    assert "的" not in jieba_result[0]


@pytest.mark.asyncio
async def test_multiple_texts():
    """Test multiple texts at once."""

    texts = [
        "我爱北京天安门",
        "I love Beijing very much",
        "今天学习 machine learning",
    ]
    jieba_result, regex_result = await compare_tokenizers(texts)

    assert len(jieba_result) == 3
    assert len(regex_result) == 3
    assert "北京" in jieba_result[0] or "天安门" in jieba_result[0]
    assert "我" in regex_result[0]
    assert "love" in regex_result[1]
    assert "beijing" in regex_result[1]


def test_tokenizer_lifecycle():
    """Test tokenizer start/close lifecycle."""

    async def run():
        tokenizer = JiebaTokenizer(filter_stopwords=True)

        assert not tokenizer.is_started
        assert len(tokenizer.stopwords) == 0

        await tokenizer.start()
        assert tokenizer.is_started
        assert len(tokenizer.stopwords) > 0

        await tokenizer.close()
        assert not tokenizer.is_started
        assert len(tokenizer.stopwords) == 0

    asyncio.run(run())


def test_jieba_tokenizer_requires_start():
    """JiebaTokenizer should fail clearly when used before startup."""
    tokenizer = JiebaTokenizer(filter_stopwords=False)
    try:
        tokenizer.tokenize(["hello 世界"])
    except RuntimeError as exc:
        assert "Call start() first" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when JiebaTokenizer is used before start()")
