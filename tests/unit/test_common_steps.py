"""Tests for reme common steps with only local dependencies."""

# pylint: disable=protected-access

import asyncio
import os
import tempfile
import warnings
from pathlib import Path

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.components.file_store import LocalFileStore
from reme.schema import FileLink, FileNode
from reme.steps.common.add import AddStep
from reme.steps.common.answer_judge import AnswerJudgeStep, AnswerJudgement
from reme.steps.common.context_answer import ContextAnswer, ContextAnswerStep
from reme.steps.common.health_check import _file_graph_status
from reme.steps.common.llm_demo import LLMDemoStep
from reme.steps.common.longmemeval_session_answer import LongMemEvalSessionAnswerStep
from reme.steps.common.memory_time_range import MemoryTimeRangeStep
from reme.steps.common.python_execute import PythonExecuteStep
from reme.steps.index import traverse as traverse_mod

warnings.filterwarnings("ignore", category=DeprecationWarning, module="jieba")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


def _run(coro):
    """Run an async coroutine on a fresh isolated event loop."""
    asyncio.run(coro)


def _node(path: str, links: list[tuple[str, str | None, str | None]] | None = None) -> FileNode:
    """Build a FileNode with (target_path, target_anchor, predicate) outgoing edges."""
    return FileNode(
        path=path,
        st_mtime=1.0,
        links=[FileLink(source_path=path, target_path=t, target_anchor=a, predicate=p) for t, a, p in (links or [])],
    )


async def _make_store(nodes: list[FileNode]) -> LocalFileStore:
    """LocalFileStore seeded with the given graph nodes (no files on disk)."""
    store = LocalFileStore(name="t", embedding_store="")
    await store.start()
    if nodes:
        await store.file_graph.upsert_nodes(nodes)
    return store


def _edges(step) -> list[dict]:
    return step.context.response.metadata.get("edges", [])


def test_add_step_coerces_numeric_inputs():
    """add accepts numeric strings as numbers, not string concatenation."""

    async def run():
        step = AddStep()
        resp = await step(a="1", b="2.5")
        assert resp.success is True
        assert resp.answer == "3.5"
        assert resp.metadata["result"] == 3.5
        print("✓ test_add_step_coerces_numeric_inputs passed")

    _run(run())


def test_add_step_rejects_invalid_inputs():
    """invalid add arguments should return a failed response instead of throwing or concatenating."""

    async def run():
        step = AddStep()
        resp = await step(a="one", b=2)
        assert resp.success is False
        assert "Invalid add arguments" in resp.answer
        print("✓ test_add_step_rejects_invalid_inputs passed")

    _run(run())


class _FakeAgentWrapper(BaseAgentWrapper):
    """Capture reply kwargs without calling a real model."""

    def __init__(self, response=None):
        super().__init__()
        self.last_kwargs = None
        self.last_inputs = None
        self.response = response or {"result": "ok"}

    async def reply(self, inputs, **kwargs) -> dict:
        self.last_inputs = inputs
        self.last_kwargs = kwargs
        return self.response


def test_llm_demo_always_registers_add_tool():
    """LLM demo always passes the add job as a tool."""

    async def run():
        wrapper = _FakeAgentWrapper()
        step = LLMDemoStep()
        resp = await step(query="hello", agent_wrapper=wrapper)
        assert resp.success is True
        assert wrapper.last_kwargs["job_tools"] == ["add"]
        assert "job_tools" not in resp.metadata
        print("✓ test_llm_demo_always_registers_add_tool passed")

    _run(run())


def test_memory_time_range_uses_python_execute_tool_without_structured_output():
    """memory time range extraction exposes python_execute without structured output."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": (
                    '{"thinking":"The question explicitly says after 2026-07-01.",'
                    '"start_dt":"2026-07-01"}'
                ),
            },
        )
        step = MemoryTimeRangeStep()
        resp = await step(
            question="Show memories after 2026-07-01",
            question_date="2026-07-02",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == (
            '{"thinking":"The question explicitly says after 2026-07-01.",'
            '"start_dt":"2026-07-01"}'
        )
        assert resp.metadata["memory_time_range"] == {
            "thinking": "The question explicitly says after 2026-07-01.",
            "start_dt": "2026-07-01",
        }
        assert wrapper.last_kwargs["job_tools"] == ["python_execute"]
        assert "output_schema" not in wrapper.last_kwargs
        print("✓ test_memory_time_range_uses_python_execute_tool_without_structured_output passed")

    _run(run())


def test_memory_time_range_returns_empty_object_without_precise_time():
    """empty/null structured fields are omitted from the final answer."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": "{}",
            },
        )
        step = MemoryTimeRangeStep()
        resp = await step(
            question="What did I say about vector databases?",
            question_date="2026-07-02",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == "{}"
        assert resp.metadata["memory_time_range"] == {}
        print("✓ test_memory_time_range_returns_empty_object_without_precise_time passed")

    _run(run())


def test_context_answer_uses_structured_output_schema():
    """context answer asks the wrapper for the dedicated schema."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": "ok",
                "structured_output": {
                    "thinking": "The context says Alice moved to Shanghai in 2025.",
                    "answer": "Alice moved to Shanghai in 2025.",
                },
            },
        )
        step = ContextAnswerStep()
        resp = await step(
            query="When did Alice move to Shanghai?",
            session_context="Alice moved to Shanghai in 2025.",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == (
            '{"thinking":"The context says Alice moved to Shanghai in 2025.",'
            '"answer":"Alice moved to Shanghai in 2025."}'
        )
        assert resp.metadata["context_answer"] == {
            "thinking": "The context says Alice moved to Shanghai in 2025.",
            "answer": "Alice moved to Shanghai in 2025.",
        }
        assert wrapper.last_kwargs["output_schema"] is ContextAnswer
        print("✓ test_context_answer_uses_structured_output_schema passed")

    _run(run())


def test_context_answer_falls_back_to_unknown_when_answer_missing():
    """missing structured answer is normalized to unknown."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": "ok",
                "structured_output": {
                    "thinking": "The context does not mention Alice.",
                    "answer": "",
                },
            },
        )
        step = ContextAnswerStep()
        resp = await step(
            query="Where does Alice live?",
            session_context="Bob lives in Beijing.",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == '{"thinking":"The context does not mention Alice.","answer":"unknown"}'
        print("✓ test_context_answer_falls_back_to_unknown_when_answer_missing passed")

    _run(run())


def test_answer_judge_uses_structured_output_schema():
    """answer judge asks the wrapper for the dedicated schema."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": "ok",
                "structured_output": {
                    "thinking": "The agent answer matches the golden answer.",
                    "answer": True,
                },
            },
        )
        step = AnswerJudgeStep()
        resp = await step(
            query="Where does Alice live?",
            agent_answer="Alice lives in Shanghai.",
            golden_answer="Alice lives in Shanghai.",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == '{"thinking":"The agent answer matches the golden answer.","answer":true}'
        assert resp.metadata["answer_judgement"] == {
            "thinking": "The agent answer matches the golden answer.",
            "answer": True,
        }
        assert wrapper.last_kwargs["output_schema"] is AnswerJudgement
        print("✓ test_answer_judge_uses_structured_output_schema passed")

    _run(run())


def test_answer_judge_normalizes_string_boolean():
    """string boolean structured output is normalized to JSON boolean."""

    async def run():
        wrapper = _FakeAgentWrapper(
            {
                "result": "ok",
                "structured_output": {
                    "thinking": "The agent answer misses the city.",
                    "answer": "false",
                },
            },
        )
        step = AnswerJudgeStep()
        resp = await step(
            query="Where does Alice live?",
            agent_answer="Alice lives somewhere in China.",
            golden_answer="Alice lives in Shanghai.",
            agent_wrapper=wrapper,
        )
        assert resp.success is True
        assert resp.answer == '{"thinking":"The agent answer misses the city.","answer":false}'
        print("✓ test_answer_judge_normalizes_string_boolean passed")

    _run(run())


def test_python_execute_returns_printed_stdout():
    """python execute returns the printed result."""

    async def run():
        step = PythonExecuteStep()
        resp = await step(code="result = 6 * 7\nprint(result)")
        assert resp.success is True
        assert resp.answer == "42"
        assert resp.metadata["stdout"] == "42"
        assert resp.metadata["returncode"] == 0
        print("✓ test_python_execute_returns_printed_stdout passed")

    _run(run())


def test_python_execute_reports_runtime_error():
    """python execute exposes stderr when code fails."""

    async def run():
        step = PythonExecuteStep()
        resp = await step(code="raise ValueError('bad input')")
        assert resp.success is False
        assert "Python exited with code" in resp.answer
        assert "ValueError: bad input" in resp.answer
        assert resp.metadata["returncode"] != 0
        print("✓ test_python_execute_reports_runtime_error passed")

    _run(run())


def test_longmemeval_session_answer_limits_agent_to_sibling_session():
    """longmemeval session answer prompt passes only query.json and sibling session."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            query_dir = Path(tmp) / "longmemeval" / "135"
            session_dir = query_dir / "session"
            session_dir.mkdir(parents=True)
            query_path = query_dir / "query.json"
            query_path.write_text('{"question":"Where should I stay?"}', encoding="utf-8")
            resolved_query_path = query_path.resolve()
            resolved_session_dir = session_dir.resolve()

            wrapper = _FakeAgentWrapper({"result": "Try the Miami hotel."})
            step = LongMemEvalSessionAnswerStep()
            resp = await step(query_path=str(query_path), agent_wrapper=wrapper)

        assert resp.success is True
        assert resp.answer == "Try the Miami hotel."
        assert f"query_json_path: {resolved_query_path}" in wrapper.last_inputs
        assert f"session_dir: {resolved_session_dir}" in wrapper.last_inputs
        assert "Do not read, list, grep, or infer from any other file or directory" in wrapper.last_kwargs["system_prompt"]
        print("✓ test_longmemeval_session_answer_limits_agent_to_sibling_session passed")

    _run(run())


def test_file_graph_health_reports_neo4j_cached_counts():
    """Neo4j file graph health should not be reported as an empty local graph."""

    class FakeNeo4jGraph:
        """Minimal Neo4j graph stub with cached health counters."""

        is_started = True
        _driver = object()
        _uri = "bolt://example"
        _database = "neo4j"
        _n_nodes = 3
        _n_edges = 4
        _n_virtual = 1

    status = _file_graph_status(FakeNeo4jGraph())
    assert status["n_nodes"] == 3
    assert status["n_edges"] == 4
    assert status["n_virtual"] == 1
    print("✓ test_file_graph_health_reports_neo4j_cached_counts passed")


# ===========================================================================
# Direct unit tests: TraverseStep
# (LocalFileStore, no HTTP server — BFS over wikilink edges)
# ===========================================================================


def test_traverse_forward_depth_1():
    """depth=1 forward returns direct outbound neighbors."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None), ("c.md", "intro", "ref")]),
                    _node("b.md"),
                    _node("c.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="a.md", direction="forward", depth=1)
            results = _edges(step)
            paths = {r["path"] for r in results}
            assert paths == {"b.md", "c.md"}
            # The 'ref' edge should report its predicate/anchor.
            c_edge = next(r for r in results if r["path"] == "c.md")
            assert c_edge["predicate"] == "ref"
            assert c_edge["anchor"] == "intro"
            assert c_edge["via"] == "a.md"
            assert c_edge["depth"] == 1
            await store.close()
        print("✓ test_traverse_forward_depth_1 passed")

    asyncio.run(run())


def test_traverse_backward_returns_inlinks():
    """direction=backward walks inbound edges."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None)]),
                    _node("c.md", [("b.md", None, None)]),
                    _node("b.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="b.md", direction="backward", depth=1)
            results = _edges(step)
            assert {r["path"] for r in results} == {"a.md", "c.md"}
            await store.close()
        print("✓ test_traverse_backward_returns_inlinks passed")

    asyncio.run(run())


def test_traverse_depth_2_expands():
    """depth=2 traverses one hop beyond direct neighbors."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("a.md", [("b.md", None, None)]),
                    _node("b.md", [("c.md", None, None)]),
                    _node("c.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="a.md", direction="forward", depth=2)
            results = _edges(step)
            depth_map = {r["path"]: r["depth"] for r in results}
            assert depth_map.get("b.md") == 1
            assert depth_map.get("c.md") == 2
            await store.close()
        print("✓ test_traverse_depth_2_expands passed")

    asyncio.run(run())


def test_traverse_short_seed_yields_empty():
    """A short (not relative to the workspace) seed isn't resolved anymore — BFS simply
    finds no edges from a path that doesn't match any graph node."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("topics/Bob.md"),
                    _node("people/Bob.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="Bob", direction="forward", depth=1)
            payload = _edges(step)
            # No error, just empty results because "Bob" isn't a graph key.
            assert payload == []
            await store.close()
        print("✓ test_traverse_short_seed_yields_empty passed")

    asyncio.run(run())


def test_traverse_not_found_seed():
    """A seed not in the graph returns an empty list (no error)."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store([_node("a.md")])
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="topics/ghost.md", direction="forward", depth=1)
            payload = _edges(step)
            assert payload == []
            await store.close()
        print("✓ test_traverse_not_found_seed passed")

    asyncio.run(run())


def test_traverse_both_directions():
    """direction=both walks out- and in-bound; depth=1 returns one hop in each direction."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            store = await _make_store(
                [
                    _node("upstream.md", [("center.md", None, None)]),
                    _node("center.md", [("downstream.md", None, None)]),
                    _node("downstream.md"),
                ],
            )
            step = traverse_mod.TraverseStep(file_store=store)
            await step(path="center.md", direction="both", depth=1)
            results = _edges(step)
            assert {r["path"] for r in results} == {"upstream.md", "downstream.md"}
            await store.close()
        print("✓ test_traverse_both_directions passed")

    asyncio.run(run())


if __name__ == "__main__":
    print("\n=== traverse step tests ===")
    test_traverse_forward_depth_1()
    test_traverse_backward_returns_inlinks()
    test_traverse_depth_2_expands()
    test_traverse_short_seed_yields_empty()
    test_traverse_not_found_seed()
    test_traverse_both_directions()
    print("\n所有测试通过!")
