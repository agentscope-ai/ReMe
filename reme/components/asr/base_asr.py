"""Abstract component for speech recognition."""

from abc import abstractmethod

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum
from ...schema.speech import ASRResult, AudioFormat


class BaseASR(BaseComponent):
    """Transcribe in-memory audio without owning capture or file access."""

    component_type = ComponentEnum.ASR

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: AudioFormat = "wav",
        sample_rate: int | None = None,
        **kwargs,
    ) -> ASRResult:
        """Recognize a complete recording; kwargs are provider parameters."""
