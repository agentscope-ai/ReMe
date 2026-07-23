"""Shared helpers for Auto Fin analysis steps."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from ....base_step import BaseStep

_OutputT = TypeVar("_OutputT", bound=BaseModel)
_STATE_PREFIX = "auto_fin_"


def json_text(value: Any) -> str:
    """Render one prompt value as readable JSON."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


class AutoFinAnalysisStep(BaseStep):
    """Base class for one stateless Auto Fin analysis invocation."""

    def state(self, key: str, default: Any = None) -> Any:
        """Read namespaced state from the current execution."""
        assert self.context is not None
        return self.context.get(f"{_STATE_PREFIX}{key}", default)

    def set_state(self, key: str, value: Any) -> None:
        """Write namespaced state into the current execution."""
        assert self.context is not None
        self.context[f"{_STATE_PREFIX}{key}"] = value

    async def reply(
        self,
        prompt_name: str,
        model: type[_OutputT],
        **values: str,
    ) -> tuple[_OutputT | None, str]:
        """Return validated structured output or a degradable error."""
        if self.agent_wrapper is None:
            self.logger.warning(
                f"[{self.name}] analysis unavailable prompt={prompt_name}: agent_wrapper is not configured",
            )
            return None, "Auto Fin analysis requires an agent_wrapper"
        try:
            self.logger.info(
                f"[{self.name}] analysis request start prompt={prompt_name} schema={model.__name__}",
            )
            result = await self.agent_wrapper.reply(
                self.prompt_format(prompt_name, **values),
                output_schema=model,
            )
            value = result.get("structured_output")
            output = value if isinstance(value, model) else model.model_validate(value)
            self.logger.info(
                f"[{self.name}] analysis request done prompt={prompt_name} schema={model.__name__}",
            )
            return output, ""
        except Exception as exc:  # Model/provider failures degrade the pipeline, not the ledger.
            self.logger.warning(
                f"[{self.name}] {prompt_name} failed: {type(exc).__name__}: {exc}",
            )
            return None, f"{type(exc).__name__}: {exc}"
