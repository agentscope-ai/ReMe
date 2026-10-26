"""Decision model transport is optional, lazy, validated, and lifecycle managed."""

# pylint: disable=protected-access,redefined-outer-name

import asyncio
import sys
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from reme.components.as_decision_model import AgentScopeAsDecisionModel
from reme.components.as_decision_model import agentscope as provider_module


@pytest.fixture
def provider_api():
    """Use actual AgentScope schemas when the optional adapter is installed."""
    return pytest.importorskip("agentscope.classifier")


@pytest.fixture
def provider(provider_api, monkeypatch):
    """Keep the question schema real while mocking the model boundary."""
    model = AsyncMock(return_value=SimpleNamespace(model="decision-test"))
    model.client = SimpleNamespace(aclose=AsyncMock())
    factory = Mock(return_value=model)
    monkeypatch.setattr(provider_api, "JevClassifierModel", factory)
    return factory, model


def test_start_and_close_never_load_optional_dependencies(monkeypatch):
    """An enabled but unused component does not need credentials or an SDK."""
    loader = Mock(side_effect=AssertionError("Unexpected provider import"))
    monkeypatch.setattr(provider_module, "import_module", loader)

    async def run():
        component = AgentScopeAsDecisionModel()
        await component.start()
        await component.close()
        await component.close()
        assert component._model is None
        loader.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure",
    [ModuleNotFoundError("No decision model adapter"), AttributeError("No model export")],
)
def test_old_agentscope_has_actionable_error(monkeypatch, failure):
    """Older installations fail on use, not at application startup."""
    monkeypatch.setattr(provider_module, "import_module", Mock(side_effect=failure))
    component = AgentScopeAsDecisionModel()
    with pytest.raises(ImportError, match=r"reme-ai\[decision\]"):
        asyncio.run(component(state="hello", questions={"q": {"type": "binary"}}))


@pytest.mark.parametrize(
    "options",
    [{"timeout": 0}, {"timeout": float("inf")}, {"max_retries": -1}, {"retry_delay": -1}, {"model": ""}],
)
def test_transport_options_are_validated_locally(options):
    """Invalid numeric options cannot silently reach the SDK."""
    with pytest.raises(ValidationError):
        AgentScopeAsDecisionModel(**options)


@pytest.mark.parametrize(
    "state,questions",
    [
        (None, {"q": {"type": "binary"}}),
        ("  ", {"q": {"type": "binary"}}),
        (["hello"], {"q": {"type": "binary"}}),
        ({"bad": object()}, {"q": {"type": "binary"}}),
        ({"bad": float("nan")}, {"q": {"type": "binary"}}),
        ("hello", {}),
        ("hello", None),
        ("hello", {" ": {"type": "binary"}}),
    ],
)
def test_invalid_request_shape_never_loads_provider(monkeypatch, state, questions):
    """Reject malformed state and mappings before optional imports."""
    loader = Mock(side_effect=AssertionError("Unexpected provider import"))
    monkeypatch.setattr(provider_module, "import_module", loader)
    with pytest.raises(ValueError):
        asyncio.run(AgentScopeAsDecisionModel()(state=state, questions=questions))
    loader.assert_not_called()


def test_valid_questions_reuse_model_and_restart_closes_clients(provider, provider_api):
    """Concurrent calls construct one model; close and restart own each client."""
    factory, model = provider
    questions = {
        "binary": {"type": "binary", "criteria": {"true": "Urgent", "false": "Routine"}},
        "choice": {"type": "choice", "criteria": {"billing": None, "support": "Technical help"}},
        "score": {"type": "score", "criteria": ["low", "high"]},
    }

    async def run():
        loop_thread = threading.get_ident()

        def construct(**_kwargs):
            assert threading.get_ident() != loop_thread
            return model

        factory.side_effect = construct
        component = AgentScopeAsDecisionModel(
            credential={"api_key": "test-key", "base_url": ""},
            model="decision-test",
            timeout=12,
            max_retries=4,
            retry_delay=0.2,
        )
        await component.start()
        factory.assert_not_called()
        responses = await asyncio.gather(*(component(state={"text": "hello"}, questions=questions) for _ in range(3)))
        assert all(response.model == "decision-test" for response in responses)
        factory.assert_called_once()
        options = factory.call_args.kwargs
        assert options["credential"].api_key.get_secret_value() == "test-key"
        assert options["credential"].base_url is None
        assert {key: value for key, value in options.items() if key != "credential"} == {
            "model": "decision-test",
            "timeout": 12.0,
            "max_retries": 4,
            "retry_delay": 0.2,
        }
        parsed = model.await_args.kwargs["questions"]
        assert isinstance(parsed["binary"], provider_api.BinaryQuestion)
        assert isinstance(parsed["choice"], provider_api.ChoiceQuestion)
        assert isinstance(parsed["score"], provider_api.ScoreQuestion)
        await component.close()
        await component.close()
        model.client.aclose.assert_awaited_once()
        assert component._model is None
        await component(state="hello", questions={"q": provider_api.BinaryQuestion()})
        assert factory.call_count == 2
        await component.close()
        assert model.client.aclose.await_count == 2

    asyncio.run(run())


@pytest.mark.parametrize("construction_fails", [False, True])
def test_cancelled_construction_keeps_ownership_until_close(provider, construction_fails):
    """Cancellation cannot abandon a client or let shutdown overtake its worker."""
    factory, model = provider

    async def run():
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), threading.Event()

        def construct(**_kwargs):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "Construction was not released"
            if construction_fails:
                raise RuntimeError("construction failed")
            return model

        factory.side_effect = construct
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        call = asyncio.create_task(component(state="hello", questions={"q": {"type": "binary"}}))
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            with pytest.raises(RuntimeError, match="Close the decision component"):
                component.model = AsyncMock()
            for _ in range(2):
                call.cancel()
                await asyncio.sleep(0)
                assert not call.done()
            closing = asyncio.create_task(component.close())
            await asyncio.sleep(0)
            assert not closing.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call, 2)
            await asyncio.wait_for(closing, 2)
            assert component.model is None
            assert not component._owns_model
            model.assert_not_awaited()
            assert model.client.aclose.await_count == (0 if construction_fails else 1)
        finally:
            release.set()
            await asyncio.gather(call, *([closing] if closing is not None else []), return_exceptions=True)
            await component.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "question",
    [
        {"type": "unknown"},
        {"type": "choice", "criteria": {}},
        {"type": "score", "criteria": []},
        {"type": "binary", "unexpected": True},
    ],
)
def test_invalid_question_never_constructs_model(provider, question):
    """Use the provider-independent schema before touching credentials."""
    factory, model = provider
    with pytest.raises(ValidationError):
        asyncio.run(AgentScopeAsDecisionModel()(state="hello", questions={"q": question}))
    factory.assert_not_called()
    model.assert_not_awaited()


def test_missing_credentials_can_be_corrected_after_failure(provider):
    """Failure leaves no partially initialized model and permits a later call."""
    factory, _ = provider

    async def run():
        component = AgentScopeAsDecisionModel()
        try:
            with pytest.raises(ValueError, match="DECISION_MODEL_API_KEY"):
                await component(state="hello", questions={"q": {"type": "binary"}})
            factory.assert_not_called()
            assert component._model is None
            component.credential["api_key"] = "test-key"
            await component(state="hello", questions={"q": {"type": "binary"}})
        finally:
            await component.close()

    asyncio.run(run())


def test_close_waits_for_all_active_requests(provider, monkeypatch):
    """Shutdown cannot close a shared client while another call is using it."""
    _, model = provider

    async def run():
        started = asyncio.Event()
        closing = asyncio.Event()
        releases = [asyncio.Event(), asyncio.Event()]
        count = 0

        async def reply(**_kwargs):
            nonlocal count
            index = count
            count += 1
            if count == 2:
                started.set()
            await releases[index].wait()
            return SimpleNamespace(model="decision-test")

        model.side_effect = reply
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        original_close = component._close

        async def mark_close():
            closing.set()
            await original_close()

        monkeypatch.setattr(component, "_close", mark_close)
        calls = [asyncio.create_task(component(state="hello", questions={"q": {"type": "binary"}})) for _ in range(2)]
        await asyncio.wait_for(started.wait(), timeout=2)
        close = asyncio.create_task(component.close())
        await asyncio.wait_for(closing.wait(), timeout=2)
        assert not close.done()
        model.client.aclose.assert_not_awaited()
        releases[0].set()
        await asyncio.wait_for(calls[0], timeout=2)
        assert not close.done()
        model.client.aclose.assert_not_awaited()
        releases[1].set()
        await asyncio.wait_for(calls[1], timeout=2)
        await asyncio.wait_for(close, timeout=2)
        model.client.aclose.assert_awaited_once()
        assert component._model is None

    asyncio.run(run())


@pytest.mark.parametrize("failure", [TimeoutError("timed out"), asyncio.CancelledError()])
def test_failed_or_cancelled_call_does_not_block_close(provider, failure):
    """Both normal failures and task cancellation release the active-call slot."""
    _, model = provider
    model.side_effect = failure

    async def run():
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        with pytest.raises(type(failure)):
            await component(state="hello", questions={"q": {"type": "binary"}})
        await asyncio.wait_for(component.close(), timeout=2)
        model.client.aclose.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("borrowed", [False, True])
def test_cancelled_close_drains_requests_and_releases_model(provider, borrowed):
    """Shutdown defers cancellation until active calls and model cleanup finish."""
    _, model = provider

    async def run():
        started, release = asyncio.Event(), asyncio.Event()

        async def reply(**_kwargs):
            started.set()
            await release.wait()

        model.side_effect = reply
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        if borrowed:
            component.model = model
        call = asyncio.create_task(component(state="hello", questions={"q": {"type": "binary"}}))
        await asyncio.wait_for(started.wait(), 2)
        close = asyncio.create_task(component.close())
        await asyncio.sleep(0)
        try:
            for _ in range(2):
                close.cancel()
                await asyncio.sleep(0)
                assert not close.done()
                assert component.is_started
                assert component.model is model
                model.client.aclose.assert_not_awaited()
            release.set()
            await asyncio.wait_for(call, 2)
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close, 2)
            assert component.model is None
            assert not component.is_started
            assert not component._owns_model
            await component.close()
            assert model.client.aclose.await_count == (0 if borrowed else 1)
        finally:
            release.set()
            await asyncio.gather(call, close, return_exceptions=True)

    asyncio.run(run())


def test_cancelled_close_finishes_owned_client_cleanup(provider):
    """Repeated shutdown cancellation cannot interrupt the client's own close."""
    _, model = provider

    async def run():
        closing, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def close_client():
            closing.set()
            await release.wait()
            finished.set()

        model.client.aclose.side_effect = close_client
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        await component(state="hello", questions={"q": {"type": "binary"}})
        close = asyncio.create_task(component.close())
        await asyncio.wait_for(closing.wait(), 2)
        try:
            for _ in range(2):
                close.cancel()
                await asyncio.sleep(0)
                assert not close.done()
                assert component.is_started
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(close, 2)
            assert finished.is_set()
            assert component.model is None
            assert not component.is_started
            await component.close()
            model.client.aclose.assert_awaited_once()
        finally:
            release.set()
            await asyncio.gather(close, return_exceptions=True)

    asyncio.run(run())


def test_client_close_error_clears_cached_model(provider):
    """A failed close cannot leave a stale client selected on the next start."""
    factory, model = provider

    async def run():
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        await component(state="hello", questions={"q": {"type": "binary"}})
        model.client.aclose.side_effect = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            await component.close()
        assert component._model is None
        assert not component.is_started
        model.client.aclose.side_effect = None
        await component(state="hello", questions={"q": {"type": "binary"}})
        assert factory.call_count == 2
        await component.close()

    asyncio.run(run())


def test_missing_sdk_has_actionable_error(provider_api, monkeypatch):
    """The actual AgentScope constructor reports the optional dependency failure."""
    assert provider_api.JevClassifierModel is not None
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)

    async def run():
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        try:
            with pytest.raises(ImportError, match=r"reme-ai\[decision\]"):
                await component(state="hello", questions={"q": {"type": "binary"}})
            assert component._model is None
        finally:
            await component.close()

    asyncio.run(run())


def test_sdk_response_and_retry_policy_round_trip(provider_api, monkeypatch):
    """Exercise the real adapter with official SDK types and a mocked HTTP boundary."""
    sdk = pytest.importorskip("typesafe_sdk")
    client = SimpleNamespace(
        system_one=AsyncMock(
            return_value=SimpleNamespace(
                model="decision-test",
                answers={
                    "binary": sdk.NoulAnswer(type="noul", noul=0.8),
                    "choice": sdk.ChoiceAnswer(type="choice", choice="a", confidence=1, probabilities={"a": 1}),
                    "score": sdk.ScoreAnswer(
                        type="score",
                        score=0.7,
                        confidence=0.7,
                        legend={0: "low", 1: "high"},
                        probabilities={0: 0.3, 1: 0.7},
                    ),
                },
                usage=sdk.Usage(input_tokens=12, output_tokens=3),
            ),
        ),
        aclose=AsyncMock(),
    )
    factory = Mock(return_value=client)
    monkeypatch.setattr(sdk, "AsyncTypeSafeClient", factory)

    async def run():
        async with AgentScopeAsDecisionModel(
            credential={"api_key": "test-key"},
            max_retries=4,
            retry_delay=0.2,
        ) as component:
            response = await component(
                state="hello",
                questions={
                    "binary": {"type": "binary"},
                    "choice": {"type": "choice", "criteria": {"a": None}},
                    "score": {"type": "score", "criteria": ["low", "high"]},
                },
            )
            assert isinstance(response, provider_api.ClassifierResponse)
            assert response.content["binary"].probability == 0.8
            assert response.content["choice"].choice == "a"
            assert response.content["score"].score == 0.7
            assert response.usage.input_tokens == 12
            assert factory.call_args.kwargs["retry"] == sdk.RetryPolicy(max_retries=4, backoff_initial=0.2)
            assert client.system_one.await_args.kwargs["questions"]["binary"].type == "noul"
        client.aclose.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.parametrize("model_name", [None, "custom-decision-model"])
def test_adapter_uses_sdk_default_or_explicit_model(monkeypatch, model_name):
    """Model defaults belong to the provider; explicit choices pass through."""
    model = AsyncMock(return_value=SimpleNamespace(content={}, usage=None))
    model.client = SimpleNamespace(aclose=AsyncMock())
    factory = Mock(return_value=model)
    modules = {
        "agentscope.classifier": SimpleNamespace(ClassifierQuestion=Any, JevClassifierModel=factory),
        "agentscope.credential": SimpleNamespace(TypeSafeCredential=SimpleNamespace),
    }
    monkeypatch.setattr(provider_module, "import_module", modules.__getitem__)

    async def run():
        async with AgentScopeAsDecisionModel(credential={"api_key": "test-key"}, model=model_name) as component:
            await component(state="hello", questions={"q": {"type": "binary"}})
            options = factory.call_args.kwargs
            if model_name is None:
                assert "model" not in options
            else:
                assert options["model"] == model_name
        model.client.aclose.assert_awaited_once()

    asyncio.run(run())


def test_injected_model_is_borrowed_and_accepts_json_questions(provider, tmp_path):
    """Application injection shares the host instance without owning its client."""
    from reme.application import Application

    factory, model = provider

    async def run():
        app = Application(
            workspace_dir=str(tmp_path),
            enable_logo=False,
            log_to_console=False,
            log_to_file=False,
            service={"backend": "http"},
            components={"as_decision_model": {"compaction": {"backend": "agentscope"}}},
        )
        component = await app.update_component("as_decision_model", "compaction", model=model)
        assert component.model is model
        async with app:
            await component(state="hello", questions={"q": {"type": "binary"}})
        factory.assert_not_called()
        model.assert_awaited_once()
        model.client.aclose.assert_not_awaited()
        assert component.model is None

    asyncio.run(run())


def test_model_replacement_guards_owned_clients(provider):
    """Replacing an owned client requires an explicit close to prevent leaks."""
    _, model = provider

    async def run():
        component = AgentScopeAsDecisionModel(credential={"api_key": "test-key"})
        await component(state="hello", questions={"q": {"type": "binary"}})
        component.model = model
        with pytest.raises(RuntimeError, match="Close"):
            component.model = None
        await component.close()
        component.model = model
        await component.close()
        # The borrowed instance is never closed a second time.
        model.client.aclose.assert_awaited_once()
        with pytest.raises(TypeError, match="callable"):
            component.model = "model-name"

    asyncio.run(run())


def test_borrowed_model_cannot_be_replaced_while_in_use(provider):
    """Host ownership still obeys the component's in-flight request boundary."""
    _, model = provider

    async def run():
        started, release = asyncio.Event(), asyncio.Event()

        async def reply(**_kwargs):
            started.set()
            await release.wait()

        model.side_effect = reply
        component = AgentScopeAsDecisionModel()
        component.model = model
        task = asyncio.create_task(component(state="hello", questions={"q": {"type": "binary"}}))
        await asyncio.wait_for(started.wait(), 2)
        with pytest.raises(RuntimeError, match="Close"):
            component.model = None
        close = asyncio.create_task(component.close())
        await asyncio.sleep(0)
        assert not close.done()
        release.set()
        await task
        await close
        model.client.aclose.assert_not_awaited()

    asyncio.run(run())
