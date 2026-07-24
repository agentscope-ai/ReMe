"""Shared Agent helpers for Auto Fin analysis steps."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from .data import AutoFinStep


class AutoFinAgentStep(AutoFinStep):
    """Validate one structured reply from a fresh Agent session."""

    async def _reply(self, prompt_name: str, model: type[BaseModel], **values: str):
        if self.agent_wrapper is None:
            raise RuntimeError("Auto Fin analysis requires an agent_wrapper")
        prompt = self.prompt_format(prompt_name, **values)
        result = await self.agent_wrapper.reply(prompt, output_schema=model)
        if not isinstance(result, dict):
            raise TypeError("Auto Fin Agent reply must be a dictionary")
        value = result.get("structured_output")
        if value is None:
            raise ValueError(f"Auto Fin Agent returned no structured output: {self._preview(result)}")
        return value if isinstance(value, model) else model.model_validate(value)

    @staticmethod
    def _preview(value: Any, limit: int = 1000) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= limit else f"{text[:limit]}...<truncated>"

    def _required(self, key: str) -> Any:
        assert self.context is not None
        if (value := self.context.get(key)) is None:
            raise RuntimeError(f"Auto Fin data is missing: {key}")
        return value
