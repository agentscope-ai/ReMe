"""OpenAIAsLLM provider backend."""

import asyncio

from agentscope.model import ChatModelBase
from agentscope.credential import OpenAICredential

from .base_as_llm import BaseAsLLM
from ..component_registry import R


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    credential_cls = OpenAICredential
    _protect_owned_model_lifecycle = True

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
