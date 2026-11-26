"""Native asynchronous DashScope speech synthesis with incremental audio."""

import asyncio
import base64
import json
from collections.abc import AsyncIterator

from pydantic import BaseModel, Field

from ...constants import SSE_MEDIA_TYPE
from .base_tts import BaseTTS
from ...core.registry import R
from ...schema.speech import AudioFormat, TTSResult
from ...utils.dashscope_speech import DEFAULT_DASHSCOPE_BASE_URL, DashScopeSpeechClient

DEFAULT_DASHSCOPE_TTS_MODEL = "qwen-audio-3.1-tts-plus"
DEFAULT_DASHSCOPE_TTS_VOICE = "longanhuan_v3.1"


class _SynthesisOptions(BaseModel):
    """Audio encoding selected for synthesis."""

    format: AudioFormat = "mp3"
    sample_rate: int = Field(default=22050, gt=0)


@R.register("dashscope")
class DashScopeTTS(BaseTTS):
    """Synthesize Qwen-Audio speech through the native HTTP SSE protocol."""

    def __init__(
        self,
        model: str = DEFAULT_DASHSCOPE_TTS_MODEL,
        voice: str = DEFAULT_DASHSCOPE_TTS_VOICE,
        audio_format: AudioFormat = "mp3",
        sample_rate: int = 22050,
        parameters: dict | None = None,
        api_key: str = "",
        base_url: str = DEFAULT_DASHSCOPE_BASE_URL,
        timeout: float = 60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.transport = DashScopeSpeechClient(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self.voice = voice
        self.audio_options = _SynthesisOptions(format=audio_format, sample_rate=sample_rate)
        self.parameters = dict(parameters or {})

    async def _start(self) -> None:
        self.transport.start()

    async def _close(self) -> None:
        await self.transport.close()

    # The abstract method describes an iterator factory, not a coroutine.
    # pylint: disable=invalid-overridden-method
    async def stream(self, text: str, **kwargs) -> AsyncIterator[TTSResult]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        headers = {**self.transport.headers(), "X-DashScope-SSE": "enable"}
        inputs = {
            "voice": self.voice,
            **self.parameters,
            **kwargs,
            "text": text,
            "format": self.audio_options.format,
            "sample_rate": self.audio_options.sample_rate,
        }
        await self.start()
        assert self.transport.client is not None
        request_id, usage = "", {}
        async with self.transport.client.stream(
            "POST",
            "services/audio/tts/SpeechSynthesizer",
            headers=headers,
            json={"model": self.model, "input": inputs},
        ) as response:
            await self.transport.check_response(response)
            if not response.headers.get("content-type", "").startswith(SSE_MEDIA_TYPE):
                await response.aread()
                self.transport.check_error(response.json())
                raise RuntimeError("DashScope TTS did not return an audio event stream")
            lines = response.aiter_lines()
            while True:
                try:
                    async with asyncio.timeout(self.transport.options.timeout):
                        line = await anext(lines)
                except StopAsyncIteration:
                    break
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                body = json.loads(data)
                self.transport.check_error(body)
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
