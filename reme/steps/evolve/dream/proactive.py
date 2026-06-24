"""Read daily interests.yaml for proactive use."""

from ...base_step import BaseStep
from ....components import R
from ....schema import ProactiveResult
from .utils import load_yaml_topics, today, workspace_dir


@R.register("proactive_step")
class ProactiveStep(BaseStep):
    """Read ``daily/<date>/interests.yaml``."""

    def __init__(self, include_content: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.include_content = include_content

    async def execute(self):
        assert self.context is not None
        day = today(self, str(self.context.get("date", "") or ""))
        include_content = bool(self.context.get("include_content", self.include_content))
        daily = self.config_value("daily_dir")
        rel_path, abs_path = f"{daily}/{day}/interests.yaml", workspace_dir(self) / daily / day / "interests.yaml"
        result = ProactiveResult(date=day, path=rel_path)

        if not abs_path.is_file():
            result.skipped, result.summary = True, f"Skipped: interests file not found at {rel_path}"
            return self._finish(True, result)
        try:
            result.content = abs_path.read_text(encoding="utf-8") if include_content else ""
            result.topics = load_yaml_topics(abs_path)
        except Exception as e:  # noqa: BLE001
            result.error, result.summary = f"{type(e).__name__}: {e}", ""
            return self._finish(False, result)

        result.summary = f"Read {len(result.topics)} proactive topic(s) from {rel_path}"
        return self._finish(True, result)

    def _finish(self, success: bool, result: ProactiveResult):
        assert self.context is not None
        self.context.response.success = success
        self.context.response.answer = result.summary if success else f"Error: {result.error}"
        self.context.response.metadata.update(result.model_dump())
        return self.context.response
