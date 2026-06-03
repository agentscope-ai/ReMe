"""Tests for RuntimeContext."""

# pylint: disable=protected-access,missing-function-docstring

import asyncio

import pytest

from reme4.components.runtime_context import RuntimeContext
from reme4.enumeration import ChunkEnum


# -- dict-like access ---------------------------------------------------------


def test_getitem_setitem():
    ctx = RuntimeContext(foo="bar")
    assert ctx["foo"] == "bar"
    ctx["baz"] = 42
    assert ctx["baz"] == 42


def test_getitem_missing_raises():
    ctx = RuntimeContext()
    with pytest.raises(KeyError):
        _ = ctx["nope"]


def test_contains():
    ctx = RuntimeContext(a=1)
    assert "a" in ctx
    assert "b" not in ctx


def test_delitem():
    ctx = RuntimeContext(a=1)
    del ctx["a"]
    assert "a" not in ctx


def test_get_with_default():
    ctx = RuntimeContext(a=1)
    assert ctx.get("a") == 1
    assert ctx.get("b", "fallback") == "fallback"
    assert ctx.get("b") is None


def test_update_merges_and_returns_self():
    ctx = RuntimeContext(a=1)
    result = ctx.update({"b": 2, "c": 3})
    assert result is ctx
    assert ctx["b"] == 2
    assert ctx["c"] == 3


# -- from_context -------------------------------------------------------------


def test_from_context_creates_new_when_none():
    ctx = RuntimeContext.from_context(None, x=10)
    assert ctx["x"] == 10


def test_from_context_reuses_existing():
    original = RuntimeContext(a=1)
    reused = RuntimeContext.from_context(original, b=2)
    assert reused is original
    assert reused["a"] == 1
    assert reused["b"] == 2


# -- apply_mapping ------------------------------------------------------------


def test_apply_mapping_copies_values():
    ctx = RuntimeContext(src="hello")
    result = ctx.apply_mapping({"src": "dst"})
    assert result is ctx
    assert ctx["dst"] == "hello"
    assert ctx["src"] == "hello"


def test_apply_mapping_skips_missing_source():
    ctx = RuntimeContext(a=1)
    ctx.apply_mapping({"missing_key": "target"})
    assert "target" not in ctx


def test_apply_mapping_empty_is_noop():
    ctx = RuntimeContext(a=1)
    result = ctx.apply_mapping({})
    assert result is ctx


# -- streaming ----------------------------------------------------------------


def test_stream_property():
    ctx_no_queue = RuntimeContext()
    assert ctx_no_queue.stream is False

    ctx_with_queue = RuntimeContext(stream_queue=asyncio.Queue())
    assert ctx_with_queue.stream is True


def test_enqueue_raises_without_queue():
    async def run():
        ctx = RuntimeContext()
        with pytest.raises(RuntimeError, match="Stream queue not initialized"):
            await ctx._enqueue(None)

    asyncio.run(run())


def test_add_stream_string():
    async def run():
        q = asyncio.Queue()
        ctx = RuntimeContext(stream_queue=q)
        result = await ctx.add_stream_string("hello", ChunkEnum.CONTENT)
        assert result is ctx

        chunk = q.get_nowait()
        assert chunk.chunk == "hello"
        assert chunk.chunk_type == ChunkEnum.CONTENT
        assert chunk.done is False

    asyncio.run(run())


def test_add_stream_done():
    async def run():
        q = asyncio.Queue()
        ctx = RuntimeContext(stream_queue=q)
        result = await ctx.add_stream_done()
        assert result is ctx

        chunk = q.get_nowait()
        assert chunk.chunk_type == ChunkEnum.DONE
        assert chunk.done is True

    asyncio.run(run())


# -- response -----------------------------------------------------------------


def test_default_response():
    ctx = RuntimeContext()
    assert ctx.response.success is True
    assert ctx.response.answer == ""


def test_custom_response():
    from reme4.schema import Response

    resp = Response(answer="ok", success=False)
    ctx = RuntimeContext(response=resp)
    assert ctx.response is resp
    assert ctx.response.success is False


if __name__ == "__main__":
    print("\n=== RuntimeContext Tests ===")
    test_getitem_setitem()
    test_getitem_missing_raises()
    test_contains()
    test_delitem()
    test_get_with_default()
    test_update_merges_and_returns_self()
    test_from_context_creates_new_when_none()
    test_from_context_reuses_existing()
    test_apply_mapping_copies_values()
    test_apply_mapping_skips_missing_source()
    test_apply_mapping_empty_is_noop()
    test_stream_property()
    test_enqueue_raises_without_queue()
    test_add_stream_string()
    test_add_stream_done()
    test_default_response()
    test_custom_response()
    print("\n所有测试通过!")
