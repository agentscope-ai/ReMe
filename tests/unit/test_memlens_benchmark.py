"""Focused offline benchmark contracts; no external services or experiment writes."""

# pylint: disable=missing-function-docstring,protected-access
import asyncio
import base64
from importlib.metadata import EntryPoint
from io import BytesIO
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import AssistantMsg, Base64Source, DataBlock, Msg, TextBlock, ToolResultBlock, ToolResultState
from PIL import Image
import pytest

from reme.components import R
from reme.components.agent_wrapper.as_agent_wrapper import AsAgentWrapper
from reme.components.component_registry import create_application_registry
from reme.components.file_store import LocalFileStore
from reme.plugin import PluginManager, _load_plugin
from reme.schema import ApplicationConfig, Response
from reme.steps.evolve.auto_memory import _sanitize_msg_for_save
from reme.steps.file_io.read_image import ReadImageStep

from benchmark.memlens import run as memlens


@pytest.fixture(autouse=True)
def memlens_judge_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/memlens-judge/src"))


def test_candidate_declares_required_image_dependency():
    manifest = Path(__file__).resolve().parents[2] / "plugins/memlens/pyproject.toml"
    project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
    assert "pillow>=10.0.0" in project["dependencies"]


def test_memlens_subset_lengths_order_and_public_inputs(tmp_path):
    image = {"file": "photo.png", "blip_caption": "HIDDEN_CAPTION"}
    row = {
        "question_type": "knowledge_update",
        "question_subtype": "update",
        "question": "What changed?",
        "answer": "HIDDEN_GOLD",
        "old_answer": "HIDDEN_OLD",
        "question_date": "2024/05/31 (Fri) 07:58",
        "haystack_dates": ["2024/05/01 (Wed) 08:00"],
        "haystack_session_ids": ["original-session"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Before <image> after", "images": [image]},
                {"role": "assistant", "content": "Original reply", "images": [], "has_answer": True},
            ],
        ],
        "answer_session_ids": ["HIDDEN_EVIDENCE"],
    }
    rows = [{**row, "question_id": f"q_{index:03d}"} for index in range(196)]
    ids = [item["question_id"] for item in rows[:195]]
    (tmp_path / "agent_subset_195.json").write_text(json.dumps({"n_questions": 195, "question_ids": ids[::-1]}))
    for length in ("128k", "256k"):
        (tmp_path / f"dataset_{length}.json").write_text(
            json.dumps([{**item, "haystack_session_ids": [f"session-{length}"]} for item in rows]),
        )

    image_dir = tmp_path / "release_images"
    image_dir.mkdir()
    Image.new("RGB", (1, 1), (12, 34, 56)).save(image_dir / "photo.png")
    scenes = memlens.load_memlens(tmp_path, "256k")
    assert [scene["id"] for scene in scenes] == ids
    first = scenes[0]
    assert first["sessions"][0]["id"] == "session-256k"
    assert memlens.load_memlens(tmp_path, "128k")[0]["sessions"][0]["id"] == "session-128k"
    assert len(first["questions"]) == 1 and first["questions"][0]["id"] == first["id"]
    assert first["sessions"][0]["timestamp"] == "2024-05-01T08:00"
    messages = memlens.session_messages(first["sessions"][0])
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [block["type"] for block in messages[0]["content"]] == ["text", "text", "data", "text"]
    assert messages[0]["content"][0]["text"] == "Before "
    assert messages[0]["content"][1]["text"] == "[Image ID: photo]"
    assert messages[0]["content"][-1]["text"] == " after"
    assert base64.b64decode(messages[0]["content"][2]["source"]["data"]) == (image_dir / "photo.png").read_bytes()
    assert memlens.session_messages(scenes[1]["sessions"][0])[0]["id"] != messages[0]["id"]
    public = json.dumps(messages) + memlens.query_prompt(first["questions"][0], "Evidence tools")
    assert "HIDDEN" not in public and "has_answer" not in public
    rows[0]["haystack_sessions"][0][0]["images"][0]["file"] = "../private.png"
    (tmp_path / "dataset_256k.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="MemLens"):
        memlens.load_memlens(tmp_path, "256k")


def test_official_metric_normalization():
    question = {"answer": "$460.00"}
    assert memlens.score_answer(question, "<think>ignore</think>$460.00<|im_end|>") == {"sub_em": 1.0, "f1": 1.0}


@pytest.fixture(name="candidate")
def fixture_candidate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins/memlens/src"))
    monkeypatch.chdir(tmp_path)
    from reme_memlens import agentic_answer as answer, auto_memory as memory

    app = SimpleNamespace(
        registry=R,
        metadata={},
        app_config=ApplicationConfig(workspace_dir=str(tmp_path)),
        jobs={},
        components={},
    )
    wrapper = AsAgentWrapper(backend="agentscope", as_llm="", app_context=app)
    wrapper.reply = AsyncMock(return_value={"result": "answer", "session_id": "new", "usage": {"input_tokens": 7}})
    store = LocalFileStore(embedding_store="", app_context=app)
    return memory, answer, app, wrapper, store, tmp_path


def test_candidate_registration_is_application_local(candidate):
    memory, answer, *_ = candidate
    plugin = _load_plugin("memlens", EntryPoint(name="memlens", value="reme_memlens", group="reme.plugins"))
    registry = create_application_registry()
    PluginManager([plugin]).register(registry)
    assert registry.get("step", "memlens_auto_memory_step") is memory.MemLensAutoMemoryStep
    assert registry.get("step", "memlens_agentic_answer_step") is answer.MemLensAgenticAnswerStep


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [True, False])
async def test_memory_saves_originals_and_native_image_links(candidate, sources):
    memory, _, app, wrapper, store, workspace = candidate
    step = memory.MemLensAutoMemoryStep(app_context=app, agent_wrapper=wrapper, file_store=store)
    step._list_session_note = AsyncMock(return_value=None)
    picture = BytesIO()
    Image.new("RGB", (1, 1), (12, 34, 56)).save(picture, format="PNG")
    image_bytes = picture.getvalue()
    image_data = base64.b64encode(image_bytes).decode()
    late = Msg.model_validate(
        {
            "id": "late",
            "name": "Alice",
            "role": "user",
            "created_at": "2026-01-02T10:00:00",
            "content": [
                TextBlock(text="Image ID: image-7"),
                DataBlock(source=Base64Source(media_type="image/png", data=image_data)),
                TextBlock(text="Before.\u2028\u2028After.\u2029End.\u0085Done."),
            ],
        },
    )
    early = late.model_copy(update={"id": "early", "created_at": "2026-01-01T10:00:00"}, deep=True)
    before = late.model_dump()
    await step(messages=[late, early], session_id="one", include_images=True, include_source_sessions=sources)
    saved = workspace / "session/dialog/one.jsonl"
    assert saved.read_text() == "".join(_sanitize_msg_for_save(msg).model_dump_json() + "\n" for msg in [early, late])
    inputs = wrapper.reply.call_args.args[0]
    assert [block.type for block in inputs.content] == ["text", "data", "text", "data", "text"]
    assert [block.source.data for block in inputs.content if isinstance(block, DataBlock)] == [image_data] * 2
    text = inputs.get_text_content()
    assert "Image ID: image-7" in text and "Before.\u2028\u2028After.\u2029End.\u0085Done." in text
    assert ("[L2 | Alice" in text) is sources
    files = sorted((workspace / "session/images/one").glob("*"))
    assert len(files) == 2 and all(path.read_bytes() == image_bytes for path in files)
    assert all(f"[[{path.relative_to(workspace).as_posix()}]]" in text for path in files)
    assert late.model_dump() == before


@pytest.mark.asyncio
async def test_read_image_returns_real_formatter_image_and_rejects_escape(candidate):
    _, answer, app, wrapper, store, workspace = candidate
    path = workspace / "session/images/one/photo.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (12, 34, 56)).save(path)

    async def native_read(**kwargs):
        step = ReadImageStep(app_context=app)
        await step(**kwargs)
        return step.context.response

    app.jobs["read_image"] = AsyncMock(side_effect=native_read)
    step = answer.MemLensAgenticAnswerStep(app_context=app, agent_wrapper=wrapper, file_store=store)
    tool = answer.image_reader(step)
    result = await tool.call(path="session/images/one/photo.png")
    assert result.state == ToolResultState.SUCCESS and isinstance(result.content[1], DataBlock)
    assert base64.b64decode(result.content[1].source.data) == path.read_bytes()
    formatted = await OpenAIChatFormatter().format(
        [
            AssistantMsg(name="tool", content=[ToolResultBlock(id="call", name="read_image", output=result.content)]),
        ],
    )
    assert any(
        block["type"] == "image_url"
        for msg in formatted
        if isinstance(msg.get("content"), list)
        for block in msg["content"]
    )
    assert (await tool.call(path="../other-scene.png")).state == ToolResultState.ERROR
    assert (await tool.call(path="mem_session/private.png")).state == ToolResultState.ERROR
    app.jobs["read_image"].assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [True, False])
async def test_answer_has_fresh_state_bm25_scope_and_no_gold(candidate, sources):
    _, answer, app, wrapper, store, workspace = candidate
    for relative in ["daily/note.md", "digest/note.md", "session/dialog/one.jsonl", "mem_session/private.md"]:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence")
    assert answer.evidence_paths(workspace, "daily", "digest", "session", False) == ["daily/note.md", "digest/note.md"]
    calls = []

    async def search(**kwargs):
        calls.append(kwargs)
        return Response(answer="evidence")

    job = AsyncMock(side_effect=search)
    job.name, job.description, job.parameters = (
        "search",
        "search",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )
    read = AsyncMock(return_value=Response(answer="evidence"))
    read.name, read.description, read.parameters = "read", "read", {"type": "object", "properties": {}}
    app.jobs = {"search": job, "read": read}
    step = answer.MemLensAgenticAnswerStep(app_context=app, agent_wrapper=wrapper, file_store=store)
    sessions, contexts = [], []
    for _ in range(2):
        response = await step(query="Official question", include_source_sessions=sources)
        options = wrapper.reply.call_args.kwargs
        assert wrapper.reply.call_args.args[0].get_text_content() == "Official question"
        assert options["resume"] == "" and options["builtin_tools"] is False and options["job_tools"] == []
        assert options["react_config"] == {"max_iters": 30}
        sessions.append(options["session_id"])
        await (await options["toolkit"].get_tool("search")).call(query="evidence")
        assert calls[-1]["vector_weight"] == 0.0
        assert ("session/dialog/one.jsonl" in calls[-1]["search_filter"]["paths"]) is sources
        assert "mem_session/private.md" not in calls[-1]["_allowed_paths"]
        contexts.append(calls[-1]["tool_context_id"])
        assert response.metadata["usage"] == {"input_tokens": 7}
    assert len(set(sessions)) == len(set(contexts)) == 2


@pytest.mark.asyncio
async def test_image_only_pipeline_preserves_sessions_and_official_query(candidate, monkeypatch):
    import reme
    from benchmark.memlens.run import normalized_usage, total_usage

    runner = memlens
    *_, workspace = candidate

    query = {
        "id": "q",
        "text": "What changed?",
        "category": "knowledge_update",
        "question_date": "2024/05/31 (Fri) 07:58",
        "answer": "HIDDEN_GOLD",
        "subtype": "update",
        "old_answer": "HIDDEN_OLD",
    }
    scene = {
        "id": "one",
        "questions": [query],
        "sessions": [{"id": name, "timestamp": "2024-05-01T08:00"} for name in ("first", "second")],
    }
    calls = []

    class Candidate:
        """Offline native-Job boundary; no provider calls."""

        def __init__(self, **_config):
            self.context = SimpleNamespace(
                components={
                    "agent_wrapper": {name: SimpleNamespace(session_path=workspace) for name in ("default", "bench")},
                },
            )
            self.closed = False

        async def start(self):
            pass

        async def close(self):
            self.closed = True

        async def run_job(self, name, **kwargs):
            calls.append((name, kwargs))
            return Response(answer="Retained answer", metadata={"agent_session_id": "new"})

    app = Candidate()
    judge = SimpleNamespace(run_job=AsyncMock(return_value=Response(metadata={"llm_judge_score": 1})))
    monkeypatch.setattr(reme, "Application", lambda **_: app)
    monkeypatch.setattr(runner, "app_config", lambda *_, **__: {})
    monkeypatch.setattr(runner, "session_messages", lambda session: [{"text": session["id"]}])
    monkeypatch.setattr(runner, "observe_usage", lambda _: None)
    rows, *_ = await runner.evaluate_scene(scene, 0, "candidate", workspace, judge, asyncio.Semaphore(2))
    assert [name for name, _ in calls] == ["auto_memory", "auto_memory", "index_update", "agentic_answer"]
    assert [kw["session_id"] for name, kw in calls if name == "auto_memory"] == ["first", "second"]
    for name, kwargs in calls:
        assert "HIDDEN" not in json.dumps(kwargs)
        if name != "index_update":
            assert kwargs["include_images"] is kwargs["include_source_sessions"] is True
    inputs = calls[-1][1]
    assert inputs["max_iters"] == 30 and "category" not in inputs
    assert inputs["query"] == memlens.query_prompt(query, "Use the available tools to retrieve relevant evidence.")
    assert judge.run_job.call_args.kwargs["reference"] == "HIDDEN_GOLD"
    assert rows[0]["metrics"]["llm_judge"] == 1 and app.closed
    measured = normalized_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 30},
        },
    )
    assert measured["noncached_input_tokens"] == 70
    assert total_usage([measured])["total_tokens"] == 120
    assert normalized_usage({})["cached_input_tokens"] is None


@pytest.mark.asyncio
async def test_observation_preserves_native_reply_and_rejects_only_interrupted():
    from benchmark.memlens.run import observe_usage

    result = {"result": "", "last_message": {"finished_reason": "exceed_max_iters", "error": None}}
    reply = AsyncMock(return_value=result)
    wrapper = SimpleNamespace(reply=reply)
    app = SimpleNamespace(context=SimpleNamespace(components={"agent_wrapper": {"default": wrapper}}))
    observe_usage(app)
    assert await wrapper.reply("Original input", react_config={"max_iters": 30}) is result
    reply.assert_awaited_once_with("Original input", react_config={"max_iters": 30})
    reply.return_value = {"last_message": {"finished_reason": "interrupted"}}
    with pytest.raises(RuntimeError, match="interrupted"):
        await wrapper.reply("Original input")
