"""LLM-based mock search operation for simulating search tool behavior.

This module provides a mock search operation that uses LLM to classify queries
and generate realistic search results based on query complexity levels.
"""

import asyncio
import json
import random
from typing import Dict, Any

from flowllm.core.context import C, FlowContext
from flowllm.core.enumeration.role import Role
from flowllm.core.op import BaseAsyncToolOp
from flowllm.core.schema import Message
from flowllm.core.schema import ToolCall
from loguru import logger


@C.register_op()
class LLMMockSearchOp(BaseAsyncToolOp):
    """
    Mock search operation that uses LLM to classify queries and simulate different scenarios.

    Supports three query complexity levels:
    - simple: Simple factual queries with short, direct answers
    - medium: Medium complexity queries requiring balanced performance
    - complex: Complex research queries requiring comprehensive, in-depth results

    Each scenario can be configured with:
    - success_rate: Probability of successful response (vs "Service busy" error)
    - extra_time: Extra sleep time in seconds to simulate latency
    - relevance_ratio: Probability of returning relevant results (vs random query results)
    """

    file_path: str = __file__

    def __init__(
        self,
        llm: str = "qwen3_30b_instruct",
        simple_config: Dict[str, Any] = None,
        medium_config: Dict[str, Any] = None,
        complex_config: Dict[str, Any] = None,
        seed: int = 0,
        **kwargs,
    ):
        """
        Initialize the LLM Mock Search Op.

        Args:
            llm: LLM model name to use for classification and content generation
            simple_config: Configuration for simple queries
                - success_rate: float (0-1), default 0.95
                - extra_time: float (seconds), default 0.5
                - relevance_ratio: float (0-1), default 0.98
            medium_config: Configuration for medium complexity queries
                - success_rate: float (0-1), default 0.85
                - extra_time: float (seconds), default 1.0
                - relevance_ratio: float (0-1), default 0.90
            complex_config: Configuration for complex queries
                - success_rate: float (0-1), default 0.70
                - extra_time: float (seconds), default 1.5
                - relevance_ratio: float (0-1), default 0.80
            seed: Random seed for deterministic behavior, default 0
        """
        super().__init__(llm=llm, **kwargs)

        # Set random seed for deterministic behavior
        self.seed = seed
        random.seed(self.seed)

        # Default configurations for each scenario
        self.simple_config = {
            "success_rate": 0.95,
            "extra_time": 0.5,
            "relevance_ratio": 0.98,
            "content_length": "short",
        }
        if simple_config:
            self.simple_config.update(simple_config)

        self.medium_config = {
            "success_rate": 0.85,
            "extra_time": 1.0,
            "relevance_ratio": 0.90,
            "content_length": "medium",
        }
        if medium_config:
            self.medium_config.update(medium_config)

        self.complex_config = {
            "success_rate": 0.70,
            "extra_time": 1.5,
            "relevance_ratio": 0.80,
            "content_length": "long",
        }
        if complex_config:
            self.complex_config.update(complex_config)

    def build_tool_call(self) -> ToolCall:
        """Build the tool call schema for the search operation.

        Returns:
            ToolCall object defining the search tool interface
        """
        return ToolCall(
            **{
                "description": "Use search keywords to retrieve relevant information from the internet.",
                "input_schema": {
                    "query": {
                        "type": "string",
                        "description": "search keyword or query",
                        "required": True,
                    },
                },
            },
        )

    async def classify_query(self, query: str) -> str:
        """
        Classify the query into simple, medium, or complex using LLM.

        Args:
            query: The search query to classify

        Returns:
            Classification result: "simple", "medium", or "complex"
        """
        classification_prompt = self.prompt_format(
            prompt_name="classification_prompt",
            query=query,
        )

        messages = [Message(role=Role.USER, content=classification_prompt)]

        response = await self.llm.achat(messages=messages)
        classification = response.content.strip().lower()

        # Extract classification from response
        if "simple" in classification:
            return "simple"
        elif "complex" in classification:
            return "complex"
        else:
            return "medium"

    async def generate_search_result(self, query: str, complexity: str, config: Dict[str, Any]) -> str:
        """
        Generate mock search results using LLM based on query complexity.

        Args:
            query: The search query
            complexity: Query complexity level
            config: Configuration for this complexity level

        Returns:
            Generated search result content
        """
        content_length = config["content_length"]

        generation_prompt = self.prompt_format(
            prompt_name="generation_prompt",
            query=query,
            complexity=complexity,
            content_length=content_length,
        )

        messages = [Message(role=Role.USER, content=generation_prompt)]

        response = await self.llm.achat(messages=messages)
        return response.content

    async def generate_random_result(self) -> str:
        """
        Generate a random/irrelevant search result to simulate low relevance.

        Returns:
            Random search result content
        """
        random_topics = [
            "the history of ancient civilizations",
            "modern technology trends",
            "climate change impacts",
            "space exploration achievements",
            "culinary traditions around the world",
            "evolution of music genres",
            "breakthroughs in medical science",
            "architectural wonders",
            "wildlife conservation efforts",
            "developments in artificial intelligence",
        ]

        random_query = random.choice(random_topics)
        generation_prompt = self.prompt_format(
            prompt_name="generation_prompt",
            query=random_query,
            complexity="simple",
            content_length="short",
        )

        messages = [Message(role=Role.USER, content=generation_prompt)]
        response = await self.llm.achat(messages=messages)

        return f"[Low Relevance Result]\n{response.content}"

    async def async_execute(self):
        """Execute the mock search operation.

        This method classifies the query, applies the appropriate configuration,
        simulates delays, and generates search results based on success and relevance rates.
        """
        query: str = self.context["query"]
        logger.info(f"LLMMockSearchOp processing query: {query}")

        # Step 1: Classify the query
        complexity = await self.classify_query(query)
        logger.info(f"Query classified as: {complexity}")

        # Step 2: Get configuration for this complexity
        if complexity == "simple":
            config = self.simple_config
        elif complexity == "medium":
            config = self.medium_config
        else:  # complex
            config = self.complex_config

        logger.info(f"Using config: {config}")

        # Step 3: Simulate extra time delay
        extra_time = config["extra_time"]
        await asyncio.sleep(extra_time)
        logger.info(f"Simulated extra delay: {extra_time:.2f}s")

        # Step 4: Check success rate
        if random.random() > config["success_rate"]:
            error_message = "Search service is currently busy. Please try again later."
            logger.warning(f"Simulated failure: {error_message}")
            result_dict = {
                "success": False,
                "content": error_message,
                "query": query,
                "complexity": complexity,
            }
            self.set_output(json.dumps(result_dict, ensure_ascii=False))
            return

        # Step 5: Check relevance ratio
        if random.random() > config["relevance_ratio"]:
            # Generate random/irrelevant result
            # NOTE: success=True because technically the tool executed without errors,
            # but the content is irrelevant (low quality), which should result in score=0.0 during evaluation
            logger.info("Generating low relevance result (success=True but low quality)")
            content = await self.generate_random_result()
            result_dict = {
                "success": True,  # Technical execution succeeded
                "content": content,
                "query": query,
                "complexity": complexity,
                "is_relevant": False,  # Mark as irrelevant for debugging
            }
        else:
            # Generate relevant result
            logger.info("Generating relevant result")
            content = await self.generate_search_result(query, complexity, config)
            result_dict = {
                "success": True,
                "content": content,
                "query": query,
                "complexity": complexity,
                "is_relevant": True,  # Mark as relevant for debugging
            }

        self.set_output(json.dumps(result_dict, ensure_ascii=False))


async def async_main():
    """Main function for testing the LLMMockSearchOp with various query types."""
    from reme_ai.main import ReMeApp

    async with ReMeApp():
        # Test with different query types
        test_queries = [
            "What is the capital of France?",  # Simple
            "How does quantum computing work?",  # Medium
            "Analyze the impact of artificial intelligence on global economy, employment, and society",  # Complex
        ]

        # Custom configurations for testing
        custom_simple = {
            "success_rate": 1,
            "extra_time": 0,
            "relevance_ratio": 1,
        }

        custom_medium = {
            "success_rate": 1,
            "extra_time": 0,
            "relevance_ratio": 1,
        }

        custom_complex = {
            "success_rate": 1,
            "extra_time": 0,
            "relevance_ratio": 1,
        }

        op = LLMMockSearchOp(
            simple_config=custom_simple,
            medium_config=custom_medium,
            complex_config=custom_complex,
        )

        for query in test_queries:
            print(f"\n{'=' * 80}")
            print(f"Testing query: {query}")
            print(f"{'=' * 80}")

            context = FlowContext(query=query)
            await op.async_call(context=context)
            result = json.loads(context.llm_mock_search_result)
            print(f"Success: {result['success']}")
            print(f"Query: {result['query']}")
            print(f"Complexity: {result['complexity']}")
            print(f"Content:\n{result['content']}")


if __name__ == "__main__":
    asyncio.run(async_main())
