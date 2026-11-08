"""Speech protocols, lifecycle, and the configured TTS-to-ASR Job chain."""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
import base64
import json
from contextlib import aclosing

import httpx
import pytest
from pydantic import ValidationError

from reme.application import Application
from reme.components import R
from reme.components import _dashscope_speech as transport_module
from reme.components.asr import BaseASR, DashScopeASR
from reme.components.tts import BaseTTS, DashScopeTTS
from reme.config.config_parser import _load_config
from reme.enumeration import ComponentEnum


class EventStream(httpx.AsyncByteStream):
    """Supply SSE frames and expose response cleanup at the transport boundary."""

    def __init__(self, frames, *, wait=False):
        self.frames = frames
        self.wait = wait
        self.waiting = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        for frame in self.frames:
            yield frame.encode()
        if self.wait:
            self.waiting.set()
            await asyncio.Event().wait()

    async def aclose(self):
        """Record the transport's close operation."""
        self.closed = True


def event(audio=b"", *, done=False, usage=None, **body):
    """Encode the native speech synthesis event, including its terminal frame."""
    payload = {
        "request_id": "tts-request",
        "output": {"audio": {"data": base64.b64encode(audio).decode()}, "finish_reason": "stop" if done else "null"},
        **body,
    }
    if usage is not None:
        payload["usage"] = usage
    return "event: result\n: heartbeat\ndata: " + json.dumps(payload) + "\n\n"


@pytest.fixture
def provider(monkeypatch):
    """Use real HTTP clients with a bounded, in-process fake provider."""
    responses, requests, clients = [], [], []
    client_class = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        assert responses, "Unexpected provider request"
        return responses.pop(0)

    def create_client(**kwargs):
        client = client_class(**kwargs, transport=httpx.MockTransport(handle))
        clients.append(client)
        return client

    monkeypatch.setattr(transport_module.httpx, "AsyncClient", create_client)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "speech-key")
    return responses, requests, clients


def sse_response(stream):
    """Return an SSE response backed by a cleanup-observable stream."""
    return httpx.Response(200, headers={"content-type": "text/event-stream;charset=UTF-8"}, stream=stream)


def test_abstract_contracts_and_builtin_registration():
    """Both capabilities require implementations and are discovered on import."""
    for base, concrete, kind in [
        (BaseASR, DashScopeASR, ComponentEnum.ASR),
        (BaseTTS, DashScopeTTS, ComponentEnum.TTS),
    ]:
        with pytest.raises(TypeError, match="abstract"):
            base()  # pylint: disable=abstract-class-instantiated
        assert R.get(kind, "dashscope") is concrete


@pytest.mark.asyncio
@pytest.mark.parametrize("component_class", [DashScopeASR, DashScopeTTS])
async def test_credentials_are_lazy_and_clients_follow_lifecycle(provider, monkeypatch, component_class):
    """Missing keys do not prevent startup, and closing releases each client."""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    component = component_class()
    await component.start()
    client = component.client
    try:
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            if isinstance(component, BaseASR):
                await component.transcribe(b"audio")
            else:
                await component.synthesize("hello")
        assert not provider[1]
    finally:
        await component.close()
    assert client.is_closed
    assert component.client is None
    await component.close()
    await component.start()
    assert component.client is not client
    await component.close()
    assert len(provider[2]) == 2


@pytest.mark.parametrize("options", [{"timeout": 0}, {"timeout": float("inf")}, {"base_url": ""}])
def test_invalid_transport_options_fail_locally(options):
    """Invalid timeouts and missing endpoints fail before any client exists."""
    with pytest.raises(ValidationError):
        DashScopeASR(**options)


@pytest.mark.asyncio
async def test_asr_native_payload_and_provider_metadata(provider):
    """Preserve transcription alignment while isolating provider authentication."""
    responses, requests, _ = provider
    raw = {
        "output": {"text": "Hello", "sentence": {"words": [{"text": "Hello", "begin_time": 10}]}},
        "request_id": "asr-request",
        "usage": {"input_tokens": 12},
    }
    responses.append(httpx.Response(200, json=raw))
    component = DashScopeASR(
        api_key="custom-key",
        base_url="https://workspace.example/api/v1/",
        parameters={"keep_dialect": True},
    )
    try:
        result = await component.transcribe(b"mp3-data", audio_format="mp3", sample_rate=22050, language_hints=["en"])
        assert result.text == "Hello"
        assert result.raw == raw
        assert result.usage == {"input_tokens": 12}
        assert result.request_id == "asr-request"
        request = requests[0]
        assert str(request.url) == "https://workspace.example/api/v1/services/aigc/multimodal-generation/generation"
        assert request.headers["authorization"] == "Bearer custom-key"
        body = json.loads(request.content)
        assert body["parameters"] == {
            "format": "mp3",
            "sample_rate": "22050",
            "keep_dialect": True,
            "language_hints": ["en"],
        }
        assert body["input"]["messages"][0]["content"] == [
            {
                "type": "input_audio",
                "input_audio": {"data": "data:audio/mpeg;base64," + base64.b64encode(b"mp3-data").decode()},
            },
        ]
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio,options",
    [
        (b"", {}),
        ("bad", {}),
        (b"x", {"audio_format": "invalid"}),
        (b"x", {"sample_rate": 0}),
        (b"x", {"sample_rate": True}),
        (b"x", {"sample_rate": 1.5}),
    ],
)
async def test_asr_rejects_invalid_audio_before_request(provider, audio, options):
    """Reject invalid binary inputs and audio descriptions without provider calls."""
    component = DashScopeASR()
    with pytest.raises(ValueError):
        await component.transcribe(audio, **options)
    assert component.client is None
    assert not provider[1]


@pytest.mark.asyncio
async def test_tts_fragments_completion_and_synthesize_share_contract(provider):
    """Streaming and collected results contain exactly the same encoded bytes."""
    frames = [event(b"first"), event(b"second", usage={"characters": 5}), event(done=True)]
    streams = [EventStream(frames), EventStream(frames)]
    provider[0].extend(sse_response(stream) for stream in streams)
    component = DashScopeTTS(voice="custom-voice", parameters={"rate": 1.2})
    try:
        chunks = [chunk async for chunk in component.stream("hello")]
        assert [chunk.audio for chunk in chunks] == [b"first", b"second", b""]
        assert [chunk.is_last for chunk in chunks] == [False, False, True]
        result = await component.synthesize("hello")
        assert result.audio == b"firstsecond"
        assert result.is_last
        assert result.usage == {"characters": 5}
        assert result.request_id == "tts-request"
        body = json.loads(provider[1][0].content)
        assert body == {
            "model": "qwen-audio-3.1-tts-plus",
            "input": {"voice": "custom-voice", "text": "hello", "format": "mp3", "sample_rate": 22050, "rate": 1.2},
        }
        assert all(stream.closed for stream in streams)
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", ["", "data: [DONE]\n\n"])
async def test_truncated_tts_is_not_a_success(provider, tail):
    """Transport EOF or DONE cannot turn a partial recording into a success."""
    stream = EventStream([event(b"partial"), tail])
    provider[0].append(sse_response(stream))
    component = DashScopeTTS()
    try:
        with pytest.raises(RuntimeError, match="without completion"):
            await component.synthesize("hello")
        assert stream.closed
    finally:
        await component.close()


@pytest.mark.asyncio
async def test_early_stream_close_releases_http_response(provider):
    """A consumer may stop playback without leaving a provider response open."""
    stream = EventStream([event(b"first")], wait=True)
    provider[0].append(sse_response(stream))
    component = DashScopeTTS()
    try:
        async with aclosing(component.stream("hello")) as chunks:
            assert (await anext(chunks)).audio == b"first"
        assert stream.closed
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancellation_close_response(provider, cancel):
    """Bound the whole audio response and propagate cancellation after cleanup."""
    stream = EventStream([event(b"first")], wait=True)
    provider[0].append(sse_response(stream))
    component = DashScopeTTS(timeout=10 if cancel else 0.01)
    task = asyncio.create_task(component.synthesize("hello"))
    try:
        await asyncio.wait_for(stream.waiting.wait(), 1)
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert stream.closed
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["http", "json", "sse", "bad_audio", "empty", "missing_text"])
async def test_provider_errors_are_not_empty_successes(provider, mode):
    """Surface HTTP, JSON, SSE, malformed audio, and missing output failures."""
    component = DashScopeTTS()
    if mode == "http":
        response, error = httpx.Response(401, json={"message": "Unauthorized"}), httpx.HTTPStatusError
    elif mode == "json":
        response, error = (
            httpx.Response(200, json={"code": "InvalidModel", "message": "unknown", "request_id": "error-id"}),
            RuntimeError,
        )
    elif mode == "sse":
        response, error = sse_response(EventStream([event(code="InvalidModel", message="unknown")])), RuntimeError
    elif mode == "bad_audio":
        response, error = sse_response(EventStream([event(output={"audio": {"data": "%%%"}})])), ValueError
    elif mode == "empty":
        response, error = sse_response(EventStream([event(done=True)])), RuntimeError
    else:
        component = DashScopeASR()
        response, error = httpx.Response(200, json={"output": {}}), RuntimeError
    provider[0].append(response)
    try:
        with pytest.raises(error):
            if isinstance(component, BaseASR):
                await component.transcribe(b"audio")
            else:
                await component.synthesize("hello")
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("component_class", [DashScopeASR, DashScopeTTS])
async def test_http_errors_preserve_provider_reason(provider, component_class):
    """ASR and streamed TTS failures must explain a provider's rejected input."""
    provider[0].append(
        httpx.Response(
            400,
            json={
                "code": "InvalidParameter",
                "message": "No valid speech detected",
                "request_id": "speech-request",
            },
        ),
    )
    component = component_class()
    try:
        with pytest.raises(httpx.HTTPStatusError, match="InvalidParameter: No valid speech detected: speech-request"):
            if isinstance(component, BaseASR):
                await component.transcribe(b"audio")
            else:
                await component.synthesize("hello")
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,message",
    [
        (400, "CLIENT_ERROR", "ASR_RESPONSE_HAVE_NO_WORDS"),
        (400, "ASR_RESPONSE_HAVE_NO_WORDS", "No words"),
        (200, "ASR_RESPONSE_HAVE_NO_WORDS", "No words"),
    ],
)
async def test_asr_empty_recognition_is_a_result(provider, status, code, message):
    """A provider's no-words response preserves diagnostics without failing the Job."""
    body = {
        "code": code,
        "message": message,
        "request_id": "empty-request",
        "usage": {"input_tokens": 2},
    }
    provider[0].append(httpx.Response(status, json=body))
    component = DashScopeASR()
    try:
        result = await component.transcribe(b"audio")
        assert result.text == ""
        assert result.request_id == "empty-request"
        assert result.usage == {"input_tokens": 2}
        assert result.raw == body
    finally:
        await component.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code,message",
    [
        (400, "CLIENT_ERROR", "Invalid audio format"),
        (400, "CLIENT_ERROR", "NOT_ASR_RESPONSE_HAVE_NO_WORDS"),
        (400, "InvalidParameter", "ASR_RESPONSE_HAVE_NO_WORDS"),
        (401, "CLIENT_ERROR", "ASR_RESPONSE_HAVE_NO_WORDS"),
        (500, "CLIENT_ERROR", "ASR_RESPONSE_HAVE_NO_WORDS"),
    ],
)
async def test_asr_does_not_suppress_other_provider_errors(provider, status, code, message):
    """Empty-recognition handling must not hide authentication or service failures."""
    provider[0].append(httpx.Response(status, json={"code": code, "message": message}))
    component = DashScopeASR()
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await component.transcribe(b"audio")
    finally:
        await component.close()


@pytest.mark.asyncio
async def test_proactive_speech_jobs_roundtrip_and_independent_calls(provider, tmp_path):
    """Load real Job definitions and pass generated bytes across fresh Steps."""
    config = _load_config("proactive.yaml")
    jobs = {name: config["jobs"][name] for name in ["speech_demo", "tts_demo", "asr_demo"]}
    app = Application(
        workspace_dir=str(tmp_path),
        components={name: config["components"][name] for name in ["asr", "tts"]},
        jobs=jobs,
        service={"backend": "cli"},
        enable_logo=False,
        log_to_file=False,
    )
    await app.start()
    try:
        provider[0].extend(
            [
                sse_response(EventStream([event(b"generated-audio"), event(done=True)])),
                httpx.Response(200, json={"output": {"text": "Hello world"}, "request_id": "recognized"}),
            ],
        )
        response = await app.run_job("speech_demo", text="Hello world")
        assert response.success, response.answer
        assert response.answer == "Hello world"
        assert response.metadata["tts"]["size_bytes"] == len(b"generated-audio")
        assert response.metadata["asr"]["request_id"] == "recognized"
        body = json.loads(provider[1][-1].content)
        assert body["parameters"] == {"format": "mp3", "sample_rate": "22050"}
        data = body["input"]["messages"][0]["content"][0]["input_audio"]["data"]
        assert base64.b64decode(data.split(",", 1)[1]) == b"generated-audio"
        provider[0].append(sse_response(EventStream([event(b"single"), event(done=True)])))
        response = await app.run_job("tts_demo", text="hello")
        assert response.success
        assert base64.b64decode(response.answer["audio"]) == b"single"
        provider[0].append(httpx.Response(200, json={"output": {"text": "standalone"}}))
        response = await app.run_job("asr_demo", audio=base64.b64encode(b"recorded").decode(), audio_format="wav")
        assert response.success
        assert response.answer == "standalone"
        provider[0].append(
            httpx.Response(
                400,
                json={
                    "code": "CLIENT_ERROR",
                    "message": "ASR_RESPONSE_HAVE_NO_WORDS",
                    "request_id": "empty-request",
                },
            ),
        )
        response = await app.run_job("asr_demo", audio=base64.b64encode(b"silent").decode(), audio_format="wav")
        assert response.success
        assert response.answer == ""
        assert response.metadata["asr"]["request_id"] == "empty-request"
        response = await app.run_job("asr_demo", audio="%%%")
        assert not response.success
    finally:
        await app.close()
    assert all(client.is_closed for client in provider[2])
