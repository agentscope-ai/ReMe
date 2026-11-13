"""Native DashScope recognition of complete audio recordings."""

import asyncio
import base64
import re

from .base_asr import BaseASR
from .._dashscope_speech import DashScopeSpeechComponent
from ..component_registry import R
from ...schema.speech import ASRResult, AudioFormat


@R.register("dashscope")
class DashScopeASR(DashScopeSpeechComponent, BaseASR):
    """Use Qwen-Audio-ASR over the native HTTP API."""

    def __init__(self, model: str = "qwen-audio-3.1-asr-flash", parameters: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.parameters = dict(parameters or {})

    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: AudioFormat = "wav",
        sample_rate: int | None = None,
        **kwargs,
    ) -> ASRResult:
        if not isinstance(audio, bytes) or not audio:
            raise ValueError("audio must be non-empty bytes")
        if audio_format not in ("wav", "mp3", "pcm", "opus"):
            raise ValueError(f"Unsupported audio format: {audio_format}")
        if sample_rate is not None and (
            not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0
        ):
            raise ValueError("sample_rate must be positive")
        headers = self._headers()
        parameters = {**self.parameters, **kwargs, "format": audio_format}
        if sample_rate is not None:
            parameters["sample_rate"] = str(sample_rate)
        mime = "audio/mpeg" if audio_format == "mp3" else f"audio/{audio_format}"
        data = f"data:{mime};base64,{base64.b64encode(audio).decode('ascii')}"
        await self.start()
        assert self.client is not None
        async with asyncio.timeout(self.options.timeout):
            response = await self.client.post(
                "services/aigc/multimodal-generation/generation",
                headers=headers,
                json={
                    "model": self.model,
                    "input": {
                        "messages": [
                            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data}}]},
                        ],
                    },
                    "parameters": parameters,
                },
            )
            try:
                body = response.json()
            except ValueError:
                await self._check_response(response)
                raise
            # Only this explicit no-words response is a normal empty transcription.
            no_words = (
                response.status_code in (200, 400)
                and isinstance(body, dict)
                and (
                    body.get("code") == "ASR_RESPONSE_HAVE_NO_WORDS"
                    or (
                        body.get("code") == "CLIENT_ERROR"
                        and re.search(r"\bASR_RESPONSE_HAVE_NO_WORDS\b", str(body.get("message", "")))
                    )
                )
            )
            if not no_words:
                await self._check_response(response)
                self._check_error(body)
        output = body.get("output") or {}
        text = "" if no_words else output.get("text", (output.get("sentence") or {}).get("text"))
        if not isinstance(text, str):
            raise RuntimeError("DashScope ASR returned no transcription text")
        return ASRResult(
            text=text,
            model=self.model,
            request_id=body.get("request_id") or "",
            usage=body.get("usage") or {},
            raw=body,
        )
