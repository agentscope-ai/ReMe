"""Integration tests: drive Agent through LLMDemoStep + Application wiring.

Requires LLM_API_KEY (and optionally LLM_BASE_URL / LLM_MODEL_NAME) in the
environment or a .env file at the repo root. Hits the real Anthropic API.
"""

import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

INTEGRATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(INTEGRATION_DIR))

# pylint: disable=wrong-import-position
from _workspace_fixture import workspace_env  # noqa: E402

from reme.steps.common.llm_demo import LLMDemoStep  # noqa: E402


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


async def _run_basic_chat(app) -> None:
    step = LLMDemoStep(app_context=app.context)
    response = await step(
        query="What is 1 + 1? Reply with just the number.",
    )
    text = (response.answer or "").strip()
    print(f"\n[basic_chat] response: {text!r}")
    assert text, "Empty assistant response"
    assert "2" in text, f"Expected '2' in response, got: {text!r}"
    print("✓ test_llm_demo_step_basic_chat passed")


async def _run_with_tool(app) -> None:
    step = LLMDemoStep(app_context=app.context)
    response = await step(
        query="Use the add tool to compute 21 + 21 and report the result.",
        sys_prompt="Use the `add` tool whenever the user asks to add numbers.",
    )
    text = (response.answer or "").strip()
    print(f"\n[with_tool] response: {text!r}")
    assert "42" in text, f"Expected '42' in response, got: {text!r}"
    print("✓ test_llm_demo_step_with_tool passed")


async def _run_structured_output(app) -> None:
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


async def _run_structured_output_enum(app) -> None:
    step = LLMDemoStep(app_context=app.context)
    response = await step(
        query="Analyze the sentiment: 'I absolutely love this product! It exceeded all my expectations.'",
        sys_prompt="You are a sentiment analysis expert. Analyze the given text.",
        structured_model=SentimentAnalysis,
    )
    structured = response.metadata.get("structured_output")
    print(f"\n[structured_enum] result: {structured}")
    assert structured is not None, "structured_output should not be None"
    assert structured["sentiment"] == "positive", f"Expected sentiment='positive', got: {structured['sentiment']}"
    assert 0 <= structured["confidence"] <= 1, f"Confidence should be 0-1, got: {structured['confidence']}"
    assert isinstance(structured["key_phrases"], list), "key_phrases should be a list"
    assert len(structured["key_phrases"]) > 0, "key_phrases should not be empty"
    print("✓ test_llm_demo_step_structured_output_enum passed")


async def _run_all() -> None:
    with workspace_env() as env:
        app = await env.make_app()
        try:
            await _run_basic_chat(app)
            await _run_with_tool(app)
            await _run_structured_output(app)
            await _run_structured_output_enum(app)
        finally:
            await env.close_all()


if __name__ == "__main__":
    print("=== LLMDemoStep + Agent integration tests ===")
    asyncio.run(_run_all())
    print("\nAll integration tests passed!")
