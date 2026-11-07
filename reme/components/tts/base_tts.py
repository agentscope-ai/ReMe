"""Abstract component for speech synthesis."""

from abc import abstractmethod
from collections.abc import AsyncIterator
from contextlib import aclosing

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema.speech import TTSResult


class BaseTTS(BaseComponent):
    """Synthesize audio without owning playback or file access."""

    component_type = ComponentEnum.TTS

    async def synthesize(self, text: str, **kwargs) -> TTSResult:
        """Collect the same stream used by incremental consumers."""
        audio, result = [], None
        async with aclosing(self.stream(text, **kwargs)) as stream:
            async for chunk in stream:
                audio.append(chunk.audio)
                result = chunk
                if chunk.is_last:
                    break
        if result is None or not result.is_last:
            raise RuntimeError("TTS stream ended without completion")
        payload = b"".join(audio)
        if not payload:
            raise RuntimeError("TTS completed without audio")
        return result.model_copy(update={"audio": payload})

    @abstractmethod
    def stream(self, text: str, **kwargs) -> AsyncIterator[TTSResult]:
        """Yield encoded audio fragments and a final completion result.

        Consumers that stop early must close the iterator, e.g. with aclosing.
        """
