"""Speech synthesis and recognition Steps sharing in-memory audio."""

import base64

from ..base_step import BaseStep, Ref
from ...components import R
from ...components.asr import BaseASR
from ...components.tts import BaseTTS
from ...enumeration import ComponentEnum


@R.register("tts_step")
class TTSStep(BaseStep):
    """Synthesize text and publish audio for the next Step or a JSON client."""

    tts = Ref(BaseTTS, ComponentEnum.TTS)

    async def execute(self):
        assert self.context is not None
        result = await self.tts.synthesize(self.context.get("text"))
        self.context.data.update(audio=result.audio, audio_format=result.format, sample_rate=result.sample_rate)
        metadata = result.model_dump(exclude={"audio"})
        metadata["size_bytes"] = len(result.audio)
        self.context.response.metadata["tts"] = metadata
        self.context.response.answer = {**metadata, "audio": base64.b64encode(result.audio).decode("ascii")}
        return self.context.response


@R.register("asr_step")
class ASRStep(BaseStep):
    """Transcribe raw audio from a preceding Step or Base64 from a client."""

    asr = Ref(BaseASR, ComponentEnum.ASR)

    async def execute(self):
        assert self.context is not None
        audio = self.context.get("audio")
        if isinstance(audio, str):
            audio = base64.b64decode(audio, validate=True)
        result = await self.asr.transcribe(
            audio,
            audio_format=self.context.get("audio_format", "wav"),
            sample_rate=self.context.get("sample_rate"),
        )
        self.context.data["transcription"] = result.text
        self.context.response.answer = result.text
        self.context.response.metadata["asr"] = result.model_dump(exclude={"raw"})
        return self.context.response
