"""Novita-compatible LLM implementation."""

from typing import AsyncGenerator, Generator
from openai import AsyncOpenAI, OpenAI

from .openai_llm import OpenAILLM
from .openai_llm_sync import OpenAILLMSync


class NovitaLLM(OpenAILLM):
    """Asynchronous LLM client for Novita-compatible APIs."""

    def __init__(self, **kwargs):
        """Initialize the Novita async client, defaulting to Novita's API endpoint."""
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = "https://api.novita.ai/openai"
        super().__init__(**kwargs)


class NovitaLLMSync(OpenAILLMSync):
    """Synchronous LLM client for Novita-compatible APIs."""

    def __init__(self, **kwargs):
        """Initialize the Novita sync client, defaulting to Novita's API endpoint."""
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = "https://api.novita.ai/openai"
        super().__init__(**kwargs)
