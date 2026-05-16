"""Tests for Tokenizers."""

import asyncio

from reme4.components.tokenizer import JiebaTokenizer, RegexTokenizer


async def compare_tokenizers(texts: list[str], filter_stopwords: bool = False, name: str = ""):
    """Compare both tokenizers on same input."""
    jieba = JiebaTokenizer(filter_stopwords=filter_stopwords)
    regex = RegexTokenizer(filter_stopwords=filter_stopwords)

    await jieba.start()
    await regex.start()

    jieba_result = jieba.tokenize(texts)
    regex_result = regex.tokenize(texts)

    print(f"\n--- {name} ---")
    print(f"输入: {texts}")
    print(f"Jieba: {jieba_result}")
    print(f"Regex: {regex_result}")

    await jieba.close()
    await regex.close()

    return jieba_result, regex_result


def test_basic_chinese():
    """Test basic Chinese text."""

    async def run():
        jieba_result, regex_result = await compare_tokenizers(
            ["我爱北京天安门", "今天天气很好"],
            name="纯中文",
        )

        assert "北京" in jieba_result[0] or "天安门" in jieba_result[0]
        assert "我" in regex_result[0]
        print("✓ test_basic_chinese passed")

    asyncio.run(run())


def test_basic_english():
    """Test basic English text."""

    async def run():
        _, regex_result = await compare_tokenizers(
            ["I love Beijing very much"],
            name="英文",
        )

        assert "love" in regex_result[0]
        assert "beijing" in regex_result[0]
        print("✓ test_basic_english passed")

    asyncio.run(run())


def test_mixed_chinese_english():
    """Test mixed Chinese-English text."""

    async def run():
        jieba_result, regex_result = await compare_tokenizers(
            ["我用 Python 学习 machine learning 和 iPhone15 Pro。"],
            name="中英混合",
        )

        assert "python" in jieba_result[0]
        assert "python" in regex_result[0]
        print("✓ test_mixed_chinese_english passed")

    asyncio.run(run())


def test_open_example():
    """Test the 'open' example."""

    async def run():
        jieba_result, regex_result = await compare_tokenizers(
            ["我觉得open很好呀，能分好次吗？"],
            name="'open' 案例",
        )

        # open 保持完整
        assert "open" in jieba_result[0]
        assert "open" in regex_result[0]

        # Regex 中文按字拆分
        assert "我" in regex_result[0]
        assert "很" in regex_result[0]

        print("✓ test_open_example passed")

    asyncio.run(run())


def test_with_stopwords():
    """Test with stopwords filtering."""

    async def run():
        jieba_result, regex_result = await compare_tokenizers(
            ["我觉得open很好呀，能分好次吗？"],
            filter_stopwords=True,
            name="停用词过滤",
        )

        # 停用词被过滤
        assert "吗" not in jieba_result[0]
        assert "吗" not in regex_result[0]
        assert "的" not in jieba_result[0]

        print("✓ test_with_stopwords passed")

    asyncio.run(run())


def test_multiple_texts():
    """Test multiple texts at once."""

    async def run():
        texts = [
            "我爱北京天安门",
            "I love Python programming",
            "今天学习 machine learning",
        ]
        jieba_result, regex_result = await compare_tokenizers(texts, name="多个文本")

        assert len(jieba_result) == 3
        assert len(regex_result) == 3
        print("✓ test_multiple_texts passed")

    asyncio.run(run())


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

        print("✓ test_tokenizer_lifecycle passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== Tokenizer Tests ===")
    test_basic_chinese()
    test_basic_english()
    test_mixed_chinese_english()
    test_open_example()
    test_with_stopwords()
    test_multiple_texts()
    test_tokenizer_lifecycle()
    print("\n所有测试通过!")
