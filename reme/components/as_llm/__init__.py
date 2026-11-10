"""LLM model wrappers for AgentScope."""

import asyncio

from agentscope.credential import (
    AnthropicCredential,
    CredentialBase,
    DashScopeCredential,
    DeepSeekCredential,
    GeminiCredential,
    MoonshotCredential,
    OllamaCredential,
    OpenAICredential,
    XAICredential,
)
from agentscope.model import ChatModelBase

from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope chat models.

    Subclasses set ``credential_cls``. Providers are constructed on first use,
    allowing local file and search jobs to run without model credentials.
    """

    component_type = ComponentEnum.AS_LLM
    credential_cls: type[CredentialBase]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model: ChatModelBase | None = None
        self._owns_model = False
        self._closed = False
        self._closing_depth = 0

    async def start(self) -> None:
        """Start without cleaning up resources owned by another lifecycle call."""
        await super().start()
        if not self._closing_depth:
            self._closed = False

    async def _rollback_start(
        self,
        start_hook_entered: bool,
        started_owned: list[BaseComponent],
    ) -> None:
        """Clean up a pre-start model while the base lifecycle lock is held."""
        await super()._rollback_start(start_hook_entered, started_owned)
        if not start_hook_entered and self._owns_model and self._model is not None:
            try:
                await self._close()
            except BaseException:
                self.logger.exception("Failed to clean up model after startup failure")

    async def close(self) -> None:
        """Prevent lazy construction throughout started or pre-start shutdown."""
        self._closing_depth += 1
        self._closed = True
        try:
            async with self._lock:
                if not self._is_started:
                    await self._close()
                    return
            await super().close()
        finally:
            self._closed = True
            self._closing_depth -= 1

    async def _start(self) -> None:
        """Keep service startup independent of provider credentials."""

    @property
    def model(self) -> ChatModelBase:
        """Initialize on first access, including direct Step dependency resolution."""
        self.initialize_model()
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, value: ChatModelBase | None) -> None:
        """Preserve explicit model injection used by standalone consumers."""
        self._model = value
        self._owns_model = False

    def initialize_model(self) -> None:
        """Construct the configured provider once, without making a remote request."""
        if self._closed and (self._owns_model or self._model is None):
            raise RuntimeError("Start the LLM component before accessing its model after close.")
        if self._model is not None:
            return
        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))
        model_cls = credential.get_chat_model_class()
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self._model = model_cls(credential=credential, parameters=parameters, **kwargs)
        self._owns_model = True


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    credential_cls = OpenAICredential

    @BaseAsLLM.model.setter
    def model(self, value: ChatModelBase | None) -> None:
        """Reject replacement while an internally owned client is still live."""
        if value is self._model:
            return
        if self._owns_model:
            raise RuntimeError("Close the OpenAI component before replacing its owned model.")
        BaseAsLLM.model.fset(self, value)

    async def _close(self) -> None:
        """Close an internally owned async client without losing cancellation-safe cleanup."""
        if not self._owns_model or self._model is None:
            return
        cleanup = asyncio.create_task(self._release_model())
        cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                if cancellation is None:
                    cancellation = exc
        cleanup.result()
        if cancellation is not None:
            raise cancellation

    async def _release_model(self) -> None:
        """Release the owned model and clear state even when provider close fails."""
        model = self._model
        try:
            client = getattr(model, "client", None)
            if client is not None:
                await client.close()
        finally:
            self._model = None
            self._owns_model = False


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    credential_cls = AnthropicCredential


@R.register("dashscope")
class DashScopeAsLLM(BaseAsLLM):
    """DashScope chat model wrapper."""

    credential_cls = DashScopeCredential


@R.register("deepseek")
class DeepSeekAsLLM(BaseAsLLM):
    """DeepSeek chat model wrapper."""

    credential_cls = DeepSeekCredential


@R.register("gemini")
class GeminiAsLLM(BaseAsLLM):
    """Gemini chat model wrapper."""

    credential_cls = GeminiCredential


@R.register("moonshot")
class MoonshotAsLLM(BaseAsLLM):
    """Moonshot chat model wrapper."""

    credential_cls = MoonshotCredential


@R.register("ollama")
class OllamaAsLLM(BaseAsLLM):
    """Ollama chat model wrapper."""

    credential_cls = OllamaCredential


@R.register("xai")
class XAIAsLLM(BaseAsLLM):
    """xAI chat model wrapper."""

    credential_cls = XAICredential


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
    "AnthropicAsLLM",
    "DashScopeAsLLM",
    "DeepSeekAsLLM",
    "GeminiAsLLM",
    "MoonshotAsLLM",
    "OllamaAsLLM",
    "XAIAsLLM",
]
