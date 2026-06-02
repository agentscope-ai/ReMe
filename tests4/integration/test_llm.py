"""Integration tests: drive Agent through LLMDemoStep + Application wiring.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real Anthropic API.
"""

import asyncio
import os
import tempfile
from typing import Literal

from pydantic import BaseModel, Field

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
    """LLMDemoStep drives Agent through self.llm."""

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


class MathResult(BaseModel):
    """Structured output for a math computation."""

    expression: str = Field(description="The math expression that was evaluated")
    result: float = Field(description="The numeric result")
    explanation: str = Field(description="Brief explanation of the computation")


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall sentiment of the text",
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
    )
    key_phrases: list[str] = Field(
        description="Key phrases that indicate the sentiment",
    )


def test_llm_demo_step_structured_output():
    """LLMDemoStep generates structured output via generate_structured_output."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                step = LLMDemoStep(app_context=app.context)
                response = await step(
                    query="What is 15 multiplied by 7? Show your work.",
                    sys_prompt="You are a math tutor. Solve the problem step by step.",
                    structured_model=MathResult,
                )
                structured = response.metadata.get("structured_output")
                print(f"\n[structured_output] result: {structured}")
                assert structured is not None, "structured_output should not be None"
                assert "result" in structured, "structured_output should have 'result' field"
                assert structured["result"] == 105, f"Expected result=105, got: {structured['result']}"
                assert "expression" in structured, "structured_output should have 'expression' field"
                assert "explanation" in structured, "structured_output should have 'explanation' field"
                print("✓ test_llm_demo_step_structured_output passed")
            finally:
                await app.close()

    asyncio.run(run())


def test_llm_demo_step_structured_output_enum():
    """LLMDemoStep structured output with Literal/enum fields."""

    async def run():
        with tempfile.TemporaryDirectory() as tmp, _temp_chdir(tmp):
            app = await _make_app()
            try:
                step = LLMDemoStep(app_context=app.context)
                response = await step(
                    query="Analyze the sentiment: 'I absolutely love this product! It exceeded all my expectations.'",
                    sys_prompt="You are a sentiment analysis expert. Analyze the given text.",
                    structured_model=SentimentAnalysis,
                )
                structured = response.metadata.get("structured_output")
                print(f"\n[structured_enum] result: {structured}")
                assert structured is not None, "structured_output should not be None"
                assert (
                    structured["sentiment"] == "positive"
                ), f"Expected sentiment='positive', got: {structured['sentiment']}"
                assert 0 <= structured["confidence"] <= 1, f"Confidence should be 0-1, got: {structured['confidence']}"
                assert isinstance(structured["key_phrases"], list), "key_phrases should be a list"
                assert len(structured["key_phrases"]) > 0, "key_phrases should not be empty"
                print("✓ test_llm_demo_step_structured_output_enum passed")
            finally:
                await app.close()

    asyncio.run(run())


if __name__ == "__main__":
    print("=== LLMDemoStep + Agent integration tests ===")
    test_llm_demo_step_basic_chat()
    test_llm_demo_step_with_tool()
    test_llm_demo_step_structured_output()
    test_llm_demo_step_structured_output_enum()
    print("\nAll integration tests passed!")
