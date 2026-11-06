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


@pytest.mark.parametrize("injected", [False, True], ids=["owned", "borrowed"])
def test_openai_start_failure_rolls_back_owned_client(injected):
    """Startup failure closes a pre-start owned client but never a borrowed one."""
    from reme.components.as_llm import OpenAIAsLLM

    class FailingOpenAIAsLLM(OpenAIAsLLM):
        async def _start(self):
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
        with pytest.raises(RuntimeError, match="start failed"):
            await llm.start()
        assert not llm.is_started
        assert llm._model is (model if injected else None)
        assert model.client.close.await_count == (0 if injected else 1)
        await llm.close()

    asyncio.run(run())
