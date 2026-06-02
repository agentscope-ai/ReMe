"""Integration tests: drive ReActAgent through LLMDemoStep + Application wiring.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real Anthropic API.
"""

import asyncio
import os
import tempfile

from reme4 import Application
from reme4.config import resolve_app_config
from reme4.steps.common.llm_demo import LLMDemoStep
from reme4.utils import load_env

load_env()


class _temp_chdir:
    """chdir to path for the duration of the block; restore on exit."""

    def __init__(self, path):
        self.path = path
        self._old = None

    def __enter__(self):
        self._old = os.getcwd()
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self._old)


async def _make_app() -> Application:
    """Build and start an Application from the default config (LLM wired via env vars)."""
    cfg = resolve_app_config(log_to_console=False, log_to_file=False, enable_logo=False)
    app = Application(**cfg)
    await app.start()
    return app


def test_llm_demo_step_basic_chat():
    """LLMDemoStep drives ReActAgent through self.llm."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                step = LLMDemoStep(app_context=app.context)
                response = await step(
                    query="What is 1 + 1? Reply with just the number.",
                )
                text = (response.answer or "").strip()
                print(f"\n[basic_chat] response: {text!r}")
                assert text, "Empty assistant response"
                assert "2" in text, f"Expected '2' in response, got: {text!r}"
                print("✓ test_llm_demo_step_basic_chat passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_llm_demo_step_with_tool():
    """LLMDemoStep registers the add tool and the agent invokes it."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                step = LLMDemoStep(app_context=app.context)
                response = await step(
                    query="Use the add tool to compute 21 + 21 and report the result.",
                    sys_prompt="Use the `add` tool whenever the user asks to add numbers.",
                    use_add_tool=True,
                )
                text = (response.answer or "").strip()
                print(f"\n[with_tool] response: {text!r}")
                assert "42" in text, f"Expected '42' in response, got: {text!r}"
                print("✓ test_llm_demo_step_with_tool passed")
            finally:
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== LLMDemoStep + ReActAgent integration tests ===")
    test_llm_demo_step_basic_chat()
    test_llm_demo_step_with_tool()
    print("\nAll integration tests passed!")
