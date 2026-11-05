"""Focused offline benchmark contracts; no external services or experiment writes."""

# pylint: disable=missing-function-docstring,protected-access
import base64
from importlib import import_module
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

from benchmark.mobilemem import run as mobilemem


def test_candidate_declares_required_image_dependency():
    manifest = Path(__file__).resolve().parents[2] / "plugins/mobilemem/pyproject.toml"
    project = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]
    assert "pillow>=10.0.0" in project["dependencies"]


@pytest.fixture(name="mobile_scene")
def fixture_mobile_scene(tmp_path):
    folder = tmp_path / "image/uid0/camera_photos"
    folder.mkdir(parents=True)
    Image.new("RGB", (3, 2), (12, 34, 56)).save(folder / "photo.png")
    sessions = [
        {
            "session_id": f"0_{index}",
            "event_start_time": f"2025-0{index}-01 09:00:00",
            "event_end_time": f"2025-0{index}-01 10:00:00",
            "dialogue": [
                {"turn": 1, "role": "user", "content_type": "text", "content": f"Original turn {index}"},
                {"turn": 2, "role": "user", "content_type": "image", "image_inline": "image/uid0/event/photo.png"},
                {"turn": 3, "role": "assistant", "content_type": "text", "content": "Original reply"},
            ],
            "dialogue_summary": "HIDDEN_SUMMARY",
            "memory_points": [{"memory_content": "HIDDEN_MEMORY"}],
        }
        for index in (2, 1)
    ]
    question = {
        "question_id": "0_q_1",
        "question": "What happened?",
        "answer": "HIDDEN_GOLD",
        "question_format": "open_ended",
        "question_type": "visual_reasoning",
        "source_session_ids": ["0_1"],
        "image_refs": ["HIDDEN_IMAGE"],
        "evidence": [{"explanation": "HIDDEN_EVIDENCE"}],
    }
    (tmp_path / "data.jsonl").write_text(json.dumps({"uuid": 0, "language": "en", "sessions": sessions}) + "\n")
    (tmp_path / "questions.jsonl").write_text(json.dumps({"uuid": 0, "questions": [question]}) + "\n")
    return tmp_path, mobilemem.load_mobilemem(tmp_path)[0]


def test_mobilemem_complete_history_ids_and_no_gold(mobile_scene):
    root, scene = mobile_scene
    assert [session["session_id"] for session in scene["sessions"]] == ["0_1", "0_2"]
    assert scene["questions"][0]["question_type"] == "visual_reasoning"
    converted = mobilemem.session_messages(scene["sessions"][0], scene["image_root"])
    assert "HIDDEN" not in json.dumps(converted)
    original = root / "image/uid0/camera_photos/photo.png"
    before = original.read_bytes()
    assert converted[1]["content"][1]["text"] == "[Image ID: image/uid0/event/photo.png]"
    assert converted[1]["content"][2]["type"] == "data"
    assert base64.b64decode(converted[1]["content"][2]["source"]["data"]) == before
    assert original.read_bytes() == before
    with pytest.raises(ValueError, match="Invalid MobileMem image path"):
        mobilemem.resolve_image(root / "image", "../outside.png", {})
    original.unlink()
    with pytest.raises(ValueError, match="Missing or ambiguous MobileMem image"):
        mobilemem.session_messages(scene["sessions"][0], scene["image_root"])


def test_official_query_and_overlap_metrics():
    assert mobilemem.query_prompt("QUESTION", "CONTEXT") == (
        "Question:\nQUESTION\nPlease answer the question based on the following memories:\nCONTEXT\n"
        + mobilemem.ANSWER_RULES
    )
    score = mobilemem.score_answer("北京 北京", "北京")
    assert score["f1"] == pytest.approx(2 / 3)
    assert score["bleu_1"] == pytest.approx(0.36787944117144233)


@pytest.fixture(name="candidate")
def fixture_candidate(tmp_path, monkeypatch):
    name = "mobilemem"
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "plugins" / name / "src"))
    monkeypatch.chdir(tmp_path)
    memory = import_module(f"reme_{name}.auto_memory")
    answer = import_module(f"reme_{name}.agentic_answer")
    prefix = "MobileMem"
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
    return name, memory, answer, prefix, app, wrapper, store, tmp_path


def test_candidate_registration_is_application_local(candidate):
    name, memory, answer, prefix, *_ = candidate
    plugin = _load_plugin(name, EntryPoint(name=name, value=f"reme_{name}", group="reme.plugins"))
    registry = create_application_registry()
    PluginManager([plugin]).register(registry)
    assert registry.get("step", f"{name}_auto_memory_step") is getattr(memory, f"{prefix}AutoMemoryStep")
    assert registry.get("step", f"{name}_agentic_answer_step") is getattr(answer, f"{prefix}AgenticAnswerStep")


@pytest.mark.asyncio
async def test_candidate_and_judge_start_with_native_settings(tmp_path, monkeypatch, mobile_scene):
    from reme import Application

    root = Path(__file__).resolve().parents[2] / "plugins"
    for package in ("mobilemem", "mobilemem-judge"):
        monkeypatch.syspath_prepend(str(root / package / "src"))
    entries = [
        EntryPoint(name="mobilemem", value="reme_mobilemem", group="reme.plugins"),
        EntryPoint(name="mobilemem-judge", value="judge_mobilemem", group="reme.plugins"),
        EntryPoint(name="mobilemem-benchmark", value="reme_mobilemem:BENCHMARK_CONFIG", group="reme.configs"),
    ]

    def installed(group, name):
        return [entry for entry in entries if entry.group == group and entry.name == name]

    monkeypatch.setattr("reme.plugin.find_entry_points", installed)
    monkeypatch.setattr("reme.config.config_parser.find_entry_points", installed)
    monkeypatch.setenv("LLM_API_KEY", "offline")
    monkeypatch.setenv("LLM_BASE_URL", "https://provider.invalid/v1")
    monkeypatch.setenv("JUDGE_MODEL_NAME", "offline-judge-override")
    candidate_app = Application(**mobilemem.candidate_config(tmp_path / "candidate", "candidate-model"))
    judge_app = Application(**mobilemem.judge_config(tmp_path / "judge"))
    try:
        await candidate_app.start()
        await judge_app.start()
        assert set(judge_app.context.jobs) == {"answer_judge"}
        assert set(judge_app.context.components) == {"as_llm"}
        audit = mobilemem.judge_settings(judge_app)
        assert audit["model"] == "offline-judge-override"
        assert audit["parameters"] == {"max_tokens": 4096, "temperature": 0, "thinking_enable": False}
        with monkeypatch.context() as environment:
            for field in ("temperature", "max_tokens", "thinking_enable"):
                environment.setenv("TEST_PRIVATE_KEY", field)
                assert mobilemem.judge_settings(judge_app)["parameters"] == audit["parameters"]
        assert audit["protocol"] == "mobilemem-omni-membase-default-judge-v1"
        assert "credential" not in json.dumps(audit) and "provider.invalid" not in json.dumps(audit)
        models = candidate_app.context.components["as_llm"]
        for component in models.values():
            assert component.kwargs["model"] == "candidate-model"
            assert component.kwargs["stream"] is False and component.kwargs["context_size"] == 1000000
            assert component.kwargs["parameters"] == {"max_tokens": 65536, "thinking_enable": True, "temperature": 0}
            assert component.kwargs["max_retries"] == 3 and component.kwargs["retry_delay"] == 60
            assert component.kwargs["client_kwargs"] == {"timeout": 600, "max_retries": 0}
        assert candidate_app.config.components["file_store"]["default"].model_dump()["embedding_store"] is None
        assert candidate_app.config.jobs["index_update"].model_dump()["watch_dirs"] == [
            "daily_dir",
            "digest_dir",
            "session_dir/dialog",
        ]
        from reme_mobilemem.auto_memory import MobileMemAutoMemoryStep

        monkeypatch.setattr(MobileMemAutoMemoryStep, "_list_session_note", AsyncMock(return_value=None))
        wrappers = candidate_app.context.components["agent_wrapper"]
        for wrapper in wrappers.values():
            wrapper.reply = AsyncMock(return_value={"result": "answer"})
        _, scene = mobile_scene
        messages = mobilemem.session_messages(scene["sessions"][0], scene["image_root"])
        for images, kwargs in ((True, {}), (False, {"include_images": False})):
            answer = await candidate_app.run_job("agentic_answer", query="Question", **kwargs)
            assert answer.success and answer.metadata["include_images"] is images
            toolkit = wrappers["bench"].reply.call_args.kwargs["toolkit"]
            assert ((await toolkit.get_tool("read_image")) is not None) is images
            memory = await candidate_app.run_job("auto_memory", messages=messages, session_id=f"s-{images}", **kwargs)
            assert memory.success
            assert isinstance(wrappers["default"].reply.call_args.args[0], Msg) is images
            assert (tmp_path / "candidate/session/images" / f"s-{images}").exists() is images
    finally:
        await judge_app.close()
        await candidate_app.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sources", [True, False])
async def test_memory_saves_originals_and_native_image_links(candidate, sources):
    _, memory, _, prefix, app, wrapper, store, workspace = candidate
    step = getattr(memory, f"{prefix}AutoMemoryStep")(app_context=app, agent_wrapper=wrapper, file_store=store)
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
    _, _, answer, prefix, app, wrapper, store, workspace = candidate
    path = workspace / "session/images/one/photo.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), (12, 34, 56)).save(path)

    async def native_read(**kwargs):
        step = ReadImageStep(app_context=app)
        await step(**kwargs)
        return step.context.response

    app.jobs["read_image"] = AsyncMock(side_effect=native_read)
    step = getattr(answer, f"{prefix}AgenticAnswerStep")(app_context=app, agent_wrapper=wrapper, file_store=store)
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
    _, _, answer, prefix, app, wrapper, store, workspace = candidate
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
    step = getattr(answer, f"{prefix}AgenticAnswerStep")(app_context=app, agent_wrapper=wrapper, file_store=store)
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
async def test_image_only_pipeline_keeps_all_sessions_and_gold_is_judge_only(mobile_scene, tmp_path, monkeypatch):
    from benchmark.mobilemem.run import normalized_usage, observe_completion

    _, scene = mobile_scene
    calls, apps = [], []

    class Candidate:
        """Offline Application stand-in preserving the native Job call boundary."""

        def __init__(self, **config):
            self.config = config
            self.closed = False
            self.context = SimpleNamespace(
                components={
                    "agent_wrapper": {name: SimpleNamespace(session_path=tmp_path) for name in ("default", "bench")},
                },
            )
            apps.append(self)

        async def start(self):
            pass

        async def close(self):
            self.closed = True

        async def run_job(self, name, **kwargs):
            calls.append((name, kwargs))
            return Response(answer="Retained answer", metadata={"agent_session_id": "new-agent"})

    judge = SimpleNamespace(
        run_job=AsyncMock(
            return_value=Response(
                metadata={"llm_judge_score": 1, "parse_default_zero": False},
            ),
        ),
    )
    monkeypatch.setattr(mobilemem, "Application", Candidate)

    def configured(workspace, model):
        return {"workspace": workspace, "model": model}

    monkeypatch.setattr(mobilemem, "candidate_config", configured)
    monkeypatch.setattr(mobilemem, "observe_usage", lambda _: None)
    monkeypatch.setattr(mobilemem, "observe_completion", lambda _: None)
    await mobilemem.evaluate_user(
        scene,
        tmp_path / "new-user",
        "candidate",
        judge,
        __import__("asyncio").Semaphore(2),
    )
    assert [name for name, _ in calls] == ["auto_memory", "auto_memory", "index_update", "agentic_answer"]
    assert [kw["session_id"] for name, kw in calls if name == "auto_memory"] == ["0_1", "0_2"]
    for name, kwargs in calls:
        assert "HIDDEN" not in json.dumps(kwargs)
        if name in {"auto_memory", "agentic_answer"}:
            assert kwargs["include_images"] is kwargs["include_source_sessions"] is True
    answer = calls[-1][1]
    assert answer["max_iters"] == 30
    assert answer["query"] == mobilemem.query_prompt(
        scene["questions"][0]["question"],
        "Use the available evidence tools.",
    )
    assert "category" not in answer and "question_type" not in answer
    judge_inputs = judge.run_job.call_args.kwargs
    assert judge_inputs["reference"] == "HIDDEN_GOLD"
    assert judge_inputs["evidence"] == ["HIDDEN_EVIDENCE"]
    result = json.loads((tmp_path / "new-user/questions/0000/result.json").read_text(encoding="utf-8"))
    assert result["metrics"]["llm_judge"] == 1 and apps[0].closed
    assert normalized_usage({"prompt_tokens": 100, "completion_tokens": 20})["cached_input_tokens"] is None
    assert (
        normalized_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 30}})[
            "noncached_input_tokens"
        ]
        == 70
    )
    raw = {"last_message": {"finished_reason": "exceed_max_iters"}, "result": "native answer"}
    native = AsyncMock(return_value=raw)
    wrapper = SimpleNamespace(reply=native)
    observe_completion(SimpleNamespace(context=SimpleNamespace(components={"agent_wrapper": {"bench": wrapper}})))
    assert await wrapper.reply("unchanged input", max_iters=30) is raw
    native.assert_awaited_once_with("unchanged input", max_iters=30)
    native.return_value = {"last_message": {"finished_reason": "interrupted"}}
    with pytest.raises(RuntimeError, match="interrupted"):
        await wrapper.reply("unchanged input")
