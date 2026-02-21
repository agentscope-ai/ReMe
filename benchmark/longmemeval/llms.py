"""LLM utilities for LongMemEval benchmark evaluation."""

import asyncio
import json
import logging
import re

from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log

from reme.core.schema import Message
from reme.core.utils import load_env
from reme.reme import ReMe

logger = logging.getLogger(__name__)

load_env()

WAIT_TIME_LOWER = 1
WAIT_TIME_UPPER = 60
RETRY_TIMES = 5


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(3),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def llm_request(reme: ReMe, prompt: str, model_name: str = "qwen3-max", **kwargs) -> str:
    """Make an LLM request using ReMe's LLM with optional model override.

    Args:
        reme: ReMe instance
        prompt: The prompt to send to the LLM
        model_name: Optional model name to override the default model (default: "qwen3-max")
        **kwargs: Additional arguments to pass to the chat method

    Returns:
        The assistant's response content
    """
    assistant_message = await reme.llm.chat(
        messages=[
            Message(role="user", content=prompt),
        ],
        model_name=model_name,
        **kwargs,
    )
    return assistant_message.content


@retry(
    wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
    stop=stop_after_attempt(RETRY_TIMES),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def llm_request_for_json(reme: ReMe, prompt: str, model_name: str = "qwen-flash", **kwargs) -> dict:
    """Make an LLM request expecting JSON response using ReMe's LLM.

    Args:
        reme: ReMe instance
        prompt: The prompt to send to the LLM
        model_name: Optional model name to override the default model (default: "qwen-flash")
        **kwargs: Additional arguments to pass to the chat method

    Returns:
        Parsed JSON object from the LLM response

    Raises:
        ValueError: If no JSON block is found in the model output
    """
    content = await llm_request(reme, prompt, model_name=model_name, **kwargs)

    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON block found in model output: {content}")

    json_str = match.group(1).strip()
    return json.loads(json_str)


if __name__ == "__main__":

    async def test():
        """Simple manual test for JSON LLM request."""
        reme = ReMe()
        await reme.start()
        try:
            r = await llm_request_for_json(
                reme,
                'hello? answer in ```json\n{"answer": "..."}```',
            )
            print(r)
        finally:
            await reme.close()

    asyncio.run(test())
