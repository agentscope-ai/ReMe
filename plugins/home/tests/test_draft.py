"""Home draft bounds and tool contracts, without tokenizer downloads."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import json

import pytest

from reme_home.agent.draft import create_draft_tools

from test_case_study import report_tokenizer  # noqa: F401  # pylint: disable=unused-import


@pytest.mark.asyncio
async def test_draft_fifo_bounds_and_shared_store():
    (add, read), store = create_draft_tools()
    assert json.loads((await read.call()).content[0].text) == {"entries": [], "entry_count": 0}
    for index in range(7):
        result = await add.call(experience=f" lesson {index} ")
        assert json.loads(result.content[0].text) == {"status": "saved"}
    assert store.read() == [f"lesson {index}" for index in range(1, 7)]
    (_other_add, other_read), same = create_draft_tools(store)
    assert same is store
    assert json.loads((await other_read.call()).content[0].text)["entries"] == store.read()
    _, fresh = create_draft_tools()
    assert fresh.read() == []
    assert all(tool.is_read_only and not tool.is_concurrency_safe for tool in (add, read))
    assert "220" not in add.description and "10%" not in add.description
    assert set(add.input_schema["properties"]) == {"experience"}


@pytest.mark.parametrize("length,expected", [(200, 200), (201, 201), (220, 220), (221, 220)])
@pytest.mark.asyncio
async def test_draft_hidden_allowance_preserves_unicode_prefix(length, expected):
    (add, _read), store = create_draft_tools()
    lesson = ("证据🙂abc" * length)[:length]
    await add.call(experience=lesson)
    assert store.read() == [lesson[:expected]]


@pytest.mark.parametrize("experience", [None, 1, "", " \n\t"])
@pytest.mark.asyncio
async def test_invalid_draft_does_not_evict(experience):
    (add, _read), store = create_draft_tools()
    await add.call(experience="Keep this lesson")
    with pytest.raises(ValueError, match="non-empty string"):
        await add.call(experience=experience)
    assert store.read() == ["Keep this lesson"]
