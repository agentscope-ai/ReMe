"""Native asynchronous DashScope speech synthesis with incremental audio."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from .base_tts import BaseTTS
from .._dashscope_speech import DashScopeSpeechComponent
from ..component_registry import R
from ...schema.speech import AudioFormat, TTSResult


class _SynthesisOptions(BaseModel):
    """Audio encoding selected for synthesis."""

    format: AudioFormat = "mp3"
    sample_rate: int = Field(default=22050, gt=0)


@R.register("dashscope")
class DashScopeTTS(DashScopeSpeechComponent, BaseTTS):
    """Synthesize Qwen-Audio speech through the native HTTP SSE protocol."""

    def __init__(
        self,
        model: str = "qwen-audio-3.1-tts-plus",
        voice: str = "longanhuan_v3.1",
        audio_format: AudioFormat = "mp3",
        sample_rate: int = 22050,
        parameters: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.voice = voice
        self.audio_options = _SynthesisOptions(format=audio_format, sample_rate=sample_rate)
        self.parameters = dict(parameters or {})

    # The abstract method describes an iterator factory, not a coroutine.
    # pylint: disable=invalid-overridden-method
    async def stream(self, text: str, **kwargs) -> AsyncIterator[TTSResult]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        headers = {**self._headers(), "X-DashScope-SSE": "enable"}
        inputs = {
            "voice": self.voice,
            **self.parameters,
            **kwargs,
            "text": text,
            "format": self.audio_options.format,
            "sample_rate": self.audio_options.sample_rate,
        }
        await self.start()
        assert self.client is not None
        request_id, usage = "", {}
        async with asyncio.timeout(self.options.timeout):
            async with self.client.stream(
                "POST",
                "services/audio/tts/SpeechSynthesizer",
                headers=headers,
                json={"model": self.model, "input": inputs},
            ) as response:
                await self._check_response(response)
                if not response.headers.get("content-type", "").startswith("text/event-stream"):
                    await response.aread()
                    self._check_error(response.json())
                    raise RuntimeError("DashScope TTS did not return an audio event stream")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    body = json.loads(data)
                    self._check_error(body)
                    request_id = body.get("request_id") or request_id
                    if body.get("usage"):
                        usage = body["usage"]
                    output = body.get("output") or {}
                    encoded = (output.get("audio") or {}).get("data") or ""
                    done = output.get("finish_reason") == "stop"
                    if encoded or done:
                        yield TTSResult(
                            audio=base64.b64decode(encoded, validate=True),
                            format=self.audio_options.format,
                            sample_rate=self.audio_options.sample_rate,
                            model=self.model,
                            request_id=request_id,
                            usage=usage,
                            is_last=done,
                        )
                    if done:
                        return
        raise RuntimeError("DashScope TTS stream ended without completion")
