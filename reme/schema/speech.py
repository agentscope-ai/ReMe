"""Provider-independent speech recognition and synthesis results."""

from typing import Literal

from pydantic import BaseModel, Field

AudioFormat = Literal["wav", "mp3", "pcm", "opus"]


class ASRResult(BaseModel):
    """Recognized text with provider usage and alignment information."""

    text: str
    model: str
    request_id: str = ""
    usage: dict = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict, repr=False)


class TTSResult(BaseModel):
    """Encoded audio, or one incremental fragment of an encoded audio stream.

    Concatenate ``audio`` bytes in order. Only ``is_last=True`` confirms
    successful completion; that final fragment may contain no audio.
    """

    audio: bytes = Field(repr=False)
    format: AudioFormat
    sample_rate: int = Field(gt=0)
    model: str
    request_id: str = ""
    usage: dict = Field(default_factory=dict)
    is_last: bool = True
