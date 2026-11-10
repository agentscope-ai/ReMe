"""Local jobs must not initialize credentialed chat providers at startup."""

# pylint: disable=protected-access

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from agentscope.model import ChatResponse

from reme.components.application_context import ApplicationContext
from reme.components.as_llm import BaseAsLLM
from reme.components.job import BaseJob
from reme.enumeration import ComponentEnum


class LazyAsLLM(BaseAsLLM):
    """Use an isolated mock credential boundary."""

    credential_cls = Mock()


def test_start_without_credentials_and_initialize_once():
    """Startup stays local; first model access preserves provider configuration."""

    async def go():
        model_cls = Mock()
        credential_cls = Mock()
        credential_cls.return_value.get_chat_model_class.return_value = model_cls
        llm = LazyAsLLM(backend="fake", model="test", parameters={"temperature": 0.5})
        llm.credential_cls = credential_cls
        await llm.start()
        assert llm._model is None
        credential_cls.assert_not_called()
        assert llm.model is model_cls.return_value
        assert llm.model is model_cls.return_value
        credential_cls.assert_called_once_with()
        model_cls.Parameters.assert_called_once_with(temperature=0.5)
        model_cls.assert_called_once_with(
            credential=credential_cls.return_value,
            parameters=model_cls.Parameters.return_value,
            model="test",
        )
        assert llm.model is model_cls.return_value
        await llm.close()

    asyncio.run(go())


def test_provider_error_is_reported_when_model_is_requested():
    """Missing credentials fail the model operation and allow a subsequent retry."""
    credential_cls = Mock(side_effect=ValueError("Missing credentials"))
    llm = LazyAsLLM(backend="fake", model="test")
    llm.credential_cls = credential_cls
    with pytest.raises(ValueError, match="Missing credentials"):
        _ = llm.model
    assert llm._model is None
    credential_cls.side_effect = None
    assert llm.model is not None


def test_compressor_job_initializes_named_model_without_wrapper(tmp_path):
    """A started job resolves the independent compressor provider on first use."""

    async def go():
        context = ApplicationContext(workspace_dir=str(tmp_path))
        model = AsyncMock(return_value=ChatResponse(content=[{"type": "text", "text": "compressed"}], is_last=True))
        credential_cls = Mock()
        model_cls = credential_cls.return_value.get_chat_model_class.return_value
        model_cls.return_value = model
        llm = LazyAsLLM(name="compressor", app_context=context, model="test")
        llm.credential_cls = credential_cls
        context.components[ComponentEnum.AS_LLM] = {"compressor": llm}
        job = BaseJob(app_context=context, steps=[{"backend": "compressor_step", "as_llm": "compressor"}])
        await llm.start()
        await job.start()
        try:
            credential_cls.assert_not_called()
            for _ in range(2):
                response = await job(text="long source text")
                assert response.success, response.answer
                assert response.answer == "compressed"
            credential_cls.assert_called_once_with()
            assert model.await_count == 2
        finally:
            await job.close()
            await llm.close()

    asyncio.run(go())


def test_explicit_model_assignment_preserves_injection():
    """An injected model bypasses provider construction."""
    llm = LazyAsLLM(model="test")
    llm.credential_cls = Mock()
    model = Mock()
    llm.model = model
    assert llm.model is model
    llm.credential_cls.assert_not_called()


@pytest.mark.asyncio
async def test_other_model_wrappers_preserve_lazy_access_after_close():
    """OpenAI lifecycle fencing must not change non-OpenAI provider semantics."""
    model_cls = Mock()
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm = LazyAsLLM(backend="fake", model="synthetic")
    llm.credential_cls = credential_cls
    await llm.start()
    model = llm.model
    await llm.close()
    assert llm.model is model
    model_cls.assert_called_once()


@pytest.mark.asyncio
async def test_owned_openai_client_closes_after_failed_call():
    """A failed model operation must not leave client cleanup until loop GC."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model_cls = Mock()
    model_cls.return_value.client.close = AsyncMock()
    model_cls.return_value.call = AsyncMock(side_effect=ConnectionError("synthetic"))
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    await llm.start()
    try:
        model = llm.model
        with pytest.raises(ConnectionError, match="synthetic"):
            await model.call("synthetic", [])
        assert model is model_cls.return_value
    finally:
        await llm.close()
    model_cls.return_value.client.close.assert_awaited_once_with()
    assert llm._model is None
    await llm.close()
    model_cls.return_value.client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_injected_openai_client_is_borrowed():
    """The component must not close externally supplied model clients."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model = Mock()
    model.client.close = AsyncMock()
    llm.model = model
    await llm.start()
    await llm.close()
    model.client.close.assert_not_awaited()
    assert llm._model is model
    assert llm.model is model


@pytest.mark.asyncio
async def test_owned_openai_client_close_failure_clears_state():
    """A failed close is propagated once without leaving ownership stranded."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model_cls = Mock()
    model_cls.return_value.client.close = AsyncMock(side_effect=RuntimeError("close failed"))
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    await llm.start()
    _ = llm.model

    with pytest.raises(RuntimeError, match="close failed"):
        await llm.close()

    assert llm._model is None
    assert not llm._owns_model
    assert not llm.is_started
    await llm.close()
    model_cls.return_value.client.close.assert_awaited_once_with()
    with pytest.raises(RuntimeError, match="after close"):
        _ = llm.model
    credential_cls.assert_called_once_with()


@pytest.mark.asyncio
async def test_repeated_cancellation_finishes_owned_openai_client_close():
    """Repeated shutdown cancellation cannot detach the provider close task."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model_cls = Mock()
    model_cls.return_value.client.close = AsyncMock()
    closing, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def close_client():
        closing.set()
        await release.wait()
        finished.set()

    model_cls.return_value.client.close.side_effect = close_client
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    await llm.start()
    model = llm.model

    close_task = asyncio.create_task(llm.close())
    await asyncio.wait_for(closing.wait(), 2)
    for message in ("first cancellation", "second cancellation", "third cancellation"):
        close_task.cancel(message)
        await asyncio.sleep(0)
        assert not close_task.done()
        assert llm._model is model
        assert llm._owns_model
        with pytest.raises(RuntimeError, match="Close the OpenAI component"):
            llm.model = Mock()

    release.set()
    with pytest.raises(asyncio.CancelledError, match="first cancellation"):
        await asyncio.wait_for(close_task, 2)
    assert finished.is_set()
    assert llm._model is None
    assert not llm._owns_model
    assert not llm.is_started
    model_cls.return_value.client.close.assert_awaited_once_with()
    assert not any(
        not task.done() and getattr(task.get_coro(), "__qualname__", "") == "OpenAIAsLLM._release_model"
        for task in asyncio.all_tasks()
    )

    await llm.close()
    model_cls.return_value.client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_pre_start_owned_openai_model_is_closed_and_restartable():
    """A model created before start is closed by an unstarted close and can restart."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    first_model, second_model = Mock(), Mock()
    first_model.client.close = AsyncMock()
    second_model.client.close = AsyncMock()
    model_cls = Mock(side_effect=[first_model, second_model])
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls

    assert llm.model is first_model
    assert not llm.is_started
    await llm.close()
    first_model.client.close.assert_awaited_once_with()
    assert llm._model is None

    with pytest.raises(RuntimeError, match="after close"):
        _ = llm.model
    await llm.close()
    first_model.client.close.assert_awaited_once_with()

    await llm.start()
    assert llm.model is second_model
    await llm.close()
    second_model.client.close.assert_awaited_once_with()
    assert model_cls.call_count == 2


def test_pre_start_borrowed_openai_model_can_be_replaced():
    """Pre-start injection and replacement remain borrowed and unclosed."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    first_model, second_model = Mock(), Mock()
    first_model.client.close = AsyncMock()
    second_model.client.close = AsyncMock()
    llm.model = first_model
    llm.model = second_model

    async def run():
        await llm.close()
        assert llm.model is second_model
        await llm.start()
        await llm.close()

    asyncio.run(run())
    first_model.client.close.assert_not_awaited()
    second_model.client.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_owned_openai_model_requires_close_before_replacement():
    """Replacement cannot abandon an internally owned client synchronously."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model_cls = Mock()
    model_cls.return_value.client.close = AsyncMock()
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    await llm.start()
    owned_model = llm.model
    borrowed_model = Mock()

    with pytest.raises(RuntimeError, match="Close the OpenAI component"):
        llm.model = borrowed_model
    assert llm._model is owned_model

    await llm.close()
    llm.model = borrowed_model
    assert llm.model is borrowed_model
    await llm.start()
    await llm.close()
    model_cls.return_value.client.close.assert_awaited_once_with()


@pytest.mark.parametrize("failure_stage", ["resolve", "hook"])
@pytest.mark.parametrize("injected", [False, True], ids=["owned", "borrowed"])
def test_openai_start_failure_rolls_back_owned_client(injected, failure_stage):
    """Startup failure closes a pre-start owned client but never a borrowed one."""
    from reme.components.as_llm import OpenAIAsLLM

    class FailingOpenAIAsLLM(OpenAIAsLLM):
        """Simulate failures at different stages of component startup."""

        async def _resolve_bindings(self):
            if failure_stage == "resolve":
                raise RuntimeError("resolve failed")
            await super()._resolve_bindings()

        async def _start(self):
            if failure_stage == "resolve":
                await super()._start()
            raise RuntimeError("start failed")

    llm = FailingOpenAIAsLLM(backend="openai", model="synthetic")
    model = Mock()
    model.client.close = AsyncMock()
    if injected:
        llm.model = model
    else:
        model_cls = Mock(return_value=model)
        credential_cls = Mock()
        credential_cls.return_value.get_chat_model_class.return_value = model_cls
        llm.credential_cls = credential_cls
        assert llm.model is model

    async def run():
        expected = "resolve failed" if failure_stage == "resolve" else "start failed"
        with pytest.raises(RuntimeError, match=expected):
            await llm.start()
        assert not llm.is_started
        assert llm._model is (model if injected else None)
        assert model.client.close.await_count == (0 if injected else 1)
        await llm.close()

    asyncio.run(run())


@pytest.mark.asyncio
async def test_owned_openai_model_cannot_be_recreated_during_shutdown():
    """No second client can appear after the first is released but close is pending."""
    from reme.components.as_llm import OpenAIAsLLM

    first_model, next_model = Mock(), Mock()
    first_model.client.close = AsyncMock()
    next_model.client.close = AsyncMock()
    model_cls = Mock(side_effect=[first_model, next_model])
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    llm.credential_cls = credential_cls

    await llm.start()
    assert llm.model is first_model
    released, finish = asyncio.Event(), asyncio.Event()
    original_release = llm._release_model

    async def release_then_pause():
        await original_release()
        released.set()
        await finish.wait()

    llm._release_model = release_then_pause
    closing = asyncio.create_task(llm.close())
    try:
        await asyncio.wait_for(released.wait(), 2)
        assert llm._model is None
        with pytest.raises(RuntimeError, match="after close"):
            _ = llm.model
        model_cls.assert_called_once()
    finally:
        finish.set()
        await asyncio.wait_for(closing, 2)

    first_model.client.close.assert_awaited_once_with()
    with pytest.raises(RuntimeError, match="after close"):
        _ = llm.model
    await llm.start()
    assert llm.model is next_model
    await llm.close()
    next_model.client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cancelling_start_waiting_for_close_does_not_close_client_twice():
    """A cancelled lifecycle-lock waiter must not clean up the closer's model."""
    from reme.components.as_llm import OpenAIAsLLM

    llm = OpenAIAsLLM(backend="openai", model="synthetic")
    model = Mock()
    closing, release = asyncio.Event(), asyncio.Event()

    async def close_client():
        closing.set()
        await release.wait()

    model.client.close = AsyncMock(side_effect=close_client)
    model_cls = Mock(return_value=model)
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    await llm.start()
    assert llm.model is model

    close_task = asyncio.create_task(llm.close())
    start_waiter = None
    try:
        await asyncio.wait_for(closing.wait(), 2)
        start_waiter = asyncio.create_task(llm.start())
        await asyncio.sleep(0)
        assert not start_waiter.done()
        start_waiter.cancel("cancel lock waiter")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(start_waiter, 1)
        model.client.close.assert_awaited_once_with()
    finally:
        release.set()
        await asyncio.wait_for(close_task, 2)
        if start_waiter is not None and not start_waiter.done():
            start_waiter.cancel()
            await asyncio.gather(start_waiter, return_exceptions=True)

    model.client.close.assert_awaited_once_with()
    assert llm._model is None
    assert not llm.is_started


@pytest.mark.asyncio
async def test_cancelled_start_waiter_cannot_cleanup_first_startup_model():
    """Cancellation before acquiring the lifecycle lock cannot own rollback."""
    from reme.components.as_llm import OpenAIAsLLM

    entered, finish = asyncio.Event(), asyncio.Event()

    class BlockingStartupOpenAI(OpenAIAsLLM):
        """Hold the first startup inside the base component lifecycle lock."""

        async def _start(self):
            entered.set()
            await finish.wait()

    llm = BlockingStartupOpenAI(backend="openai", model="synthetic")
    model = Mock()
    model.client.close = AsyncMock()
    model_cls = Mock(return_value=model)
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm.credential_cls = credential_cls
    assert llm.model is model

    first_start = asyncio.create_task(llm.start())
    second_start = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second_start = asyncio.create_task(llm.start())
        await asyncio.sleep(0)
        assert not second_start.done()
        second_start.cancel("cancel queued startup")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(second_start, 1)
        model.client.close.assert_not_awaited()
        assert llm._model is model
    finally:
        finish.set()
        await asyncio.wait_for(first_start, 2)
        if second_start is not None and not second_start.done():
            second_start.cancel()
            await asyncio.gather(second_start, return_exceptions=True)

    assert llm.is_started
    assert llm.model is model
    await llm.close()
    model.client.close.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["resolve", "hook"])
async def test_startup_rollback_blocks_lazy_owned_model_recreation(failure_stage):
    """Both failed-start paths must fence synchronous model access during cleanup."""
    from reme.components.as_llm import OpenAIAsLLM

    class FailingOpenAIAsLLM(OpenAIAsLLM):
        """Exercise failures before and within the component startup hook."""

        async def _resolve_bindings(self):
            if failure_stage == "resolve":
                raise RuntimeError("resolve failed")
            await super()._resolve_bindings()

        async def _start(self):
            raise RuntimeError("hook failed")

    first, replacement = Mock(), Mock()
    first.client.close = AsyncMock()
    replacement.client.close = AsyncMock()
    model_cls = Mock(side_effect=[first, replacement])
    credential_cls = Mock()
    credential_cls.return_value.get_chat_model_class.return_value = model_cls
    llm = FailingOpenAIAsLLM(backend="openai", model="synthetic")
    llm.credential_cls = credential_cls
    assert llm.model is first

    released, resume = asyncio.Event(), asyncio.Event()
    original_release = llm._release_model

    async def pause_after_release():
        await original_release()
        released.set()
        await resume.wait()

    llm._release_model = pause_after_release
    starting = asyncio.create_task(llm.start())
    try:
        await asyncio.wait_for(released.wait(), 2)
        assert llm._model is None
        with pytest.raises(RuntimeError, match="after close"):
            _ = llm.model
        model_cls.assert_called_once()
    finally:
        resume.set()

    expected = "resolve failed" if failure_stage == "resolve" else "hook failed"
    with pytest.raises(RuntimeError, match=expected):
        await asyncio.wait_for(starting, 2)
    first.client.close.assert_awaited_once_with()
    assert not llm.is_started
    assert llm._model is None
    assert not llm._closed
    assert llm.model is replacement
    await llm.close()
    replacement.client.close.assert_awaited_once_with()
