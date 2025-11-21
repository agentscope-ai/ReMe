"""
ReMeApp - Reflexive Memory Application

This module provides the main application class for the ReMe (Reflexive Memory) system,
which extends FlowLLM with specialized memory management capabilities including:
- Task Memory: Store and retrieve task execution histories
- Tool Memory: Track tool usage patterns and experiences
- Personal Memory: Manage user preferences and personal information
"""

import asyncio
import sys

from flowllm.core.application import Application
from flowllm.core.context import C
from flowllm.core.schema import FlowResponse

from reme_ai.config.config_parser import ConfigParser


class ReMeApp(Application):
    """
    ReMeApp - Main application class for Reflexive Memory system.

    ReMeApp extends FlowLLMApp to provide enhanced memory capabilities for AI agents.
    It manages multiple types of memories and provides both synchronous and asynchronous
    execution interfaces for memory-enhanced workflows.
    """

    def __init__(
        self,
        *args,
        llm_api_key: str = None,
        llm_api_base: str = None,
        embedding_api_key: str = None,
        embedding_api_base: str = None,
        config_path: str = None,
        **kwargs,
    ):
        """
        Initialize ReMeApp with configuration for LLM, embeddings, and vector stores.

        ⚠️ IMPORTANT: The initialization parameters here are consistent with the command-line
        startup parameters shown in README.md. You can use the same configuration in both ways:

        Command-line startup:
            ```bash
            reme \
                backend=http \
                http.port=8002 \
                llm.default.model_name=qwen3-30b-a3b-thinking-2507 \
                embedding_model.default.model_name=text-embedding-v4 \
                vector_store.default.backend=memory
            ```

        Python API equivalent:
            ```python
            app = ReMeApp(
                "llm.default.model_name=qwen3-30b-a3b-thinking-2507",
                "embedding_model.default.model_name=text-embedding-v4",
                "vector_store.default.backend=memory"
            )
            ```

        Both approaches accept the same configuration parameters and produce identical results.

        Args:
            *args: Additional command-line style arguments passed to parser.
                These parameters are identical to the command-line startup parameters in README.

                Common configuration examples:
                For complete configuration reference, see: reme_ai/config/default.yaml

                LLM Configuration:
                - "llm.default.model_name=qwen3-30b-a3b-thinking-2507" - Set LLM model
                - "llm.default.backend=openai_compatible" - Set LLM backend type
                - "llm.default.params={'temperature': '0.6'}" - Set model parameters

                Embedding Configuration:
                - "embedding_model.default.model_name=text-embedding-v4" - Set embedding model
                - "embedding_model.default.backend=openai_compatible" - Set embedding backend
                - "embedding_model.default.params={'dimensions': 1024}" - Embedding parameters

                Vector Store Configuration:
                - "vector_store.default.backend=local" - Use local vector store
                - "vector_store.default.backend=memory" - Use memory vector store
                - "vector_store.default.backend=qdrant" - Use Qdrant vector store
                - "vector_store.default.backend=elasticsearch" - Use Elasticsearch
                - "vector_store.default.embedding_model=default" - Link vector store to embedding model
                - "vector_store.default.params={'collection_name': 'my_memories'}" - Vector store parameters
            llm_api_key: API key for LLM service (e.g., OpenAI, Claude).
                If provided, this will override the FLOW_LLM_API_KEY environment variable.
                Environment variable: FLOW_LLM_API_KEY
            llm_api_base: Base URL for LLM API. Use this for custom or self-hosted endpoints.
                If provided, this will override the FLOW_LLM_BASE_URL environment variable.
                Example: "https://api.openai.com/v1"
                Environment variable: FLOW_LLM_BASE_URL
            embedding_api_key: API key for embedding service. Can be different from llm_api_key
                if using separate services for embeddings.
                If provided, this will override the FLOW_EMBEDDING_API_KEY environment variable.
                Environment variable: FLOW_EMBEDDING_API_KEY
            embedding_api_base: Base URL for embedding API. For custom embedding endpoints.
                If provided, this will override the FLOW_EMBEDDING_BASE_URL environment variable.
                Environment variable: FLOW_EMBEDDING_BASE_URL
            config_path: Path to custom configuration YAML file. If provided, loads configuration from this file.
                Example: "path/to/my_config.yaml"
                This overrides the default configuration with your custom settings.
            **kwargs: Additional keyword arguments passed to parser. Same format as args but as key-value pairs.
                Example: model_name="gpt-4", temperature=0.7

        Raises:
            AssertionError: If required configurations are missing or invalid.

        Note:
            - Parameters here mirror the command-line options in README.md exactly
            - API keys can be provided via arguments or environment variables (see example.env)
            - The parser (ConfigParser) handles merging default configs with custom overrides
            - Vector store configuration determines where memories are persisted
            - For detailed startup examples and all available parameters, refer to README.md Quick Start section

        See Also:
            - README.md "Quick Start" section for command-line startup examples
            - README.md "Environment Configuration" for environment variable setup
            - example.env for all available environment variables
        """
        super().__init__(
            *args,
            llm_api_key=llm_api_key,
            llm_api_base=llm_api_base,
            embedding_api_key=embedding_api_key,
            embedding_api_base=embedding_api_base,
            service_config=None,
            parser=ConfigParser,
            config_path=config_path,
            load_default_config=True,
            **kwargs,
        )

    async def async_execute(self, name: str, **kwargs) -> dict:
        """
        Asynchronously execute a named flow with given parameters.

        This method executes a registered flow (workflow) by name and returns the result.
        Flows are defined in the configuration and registered during app initialization.

        Args:
            name: Name of the flow to execute. Must be registered in C.flow_dict.
                Common flows in ReMe:
                - "task_memory_flow": Query and manage task memories
                - "tool_memory_flow": Retrieve tool usage experiences
                - "personal_memory_flow": Access personal preferences
                - "sop_memory_flow": Execute standard operating procedures
            **kwargs: Keyword arguments passed to the flow execution.
                Arguments vary by flow type. Common parameters:
                - query (str): User query or instruction
                - context (dict): Additional context for the flow
                - max_results (int): Maximum number of results to return
                - threshold (float): Similarity threshold for retrieval

        Returns:
            dict: Flow execution result as a dictionary containing:
                - status: Execution status (success/failure)
                - result: Flow output data
                - metadata: Additional execution metadata

        Raises:
            AssertionError: If the flow name is not registered in C.flow_dict.

        Example:
            ```python
            result = await app.async_execute(
                "task_memory_flow",
                query="Show me all Python debugging tasks",
                max_results=10
            )
            print(result['result'])
            ```
        """
        assert name in C.flow_dict, f"Invalid flow_name={name} !"
        result: FlowResponse = await self.async_execute_flow(name=name, **kwargs)
        return result.model_dump()

    def execute(self, name: str, **kwargs) -> dict:
        """
        Synchronously execute a named flow with given parameters.

        This is a convenience wrapper around async_execute() for synchronous contexts.
        It internally uses asyncio.run() to execute the async flow.

        Args:
            name: Name of the flow to execute. See async_execute() for available flows.
            **kwargs: Keyword arguments passed to the flow. See async_execute() for details.

        Returns:
            dict: Flow execution result. Same format as async_execute().

        Raises:
            AssertionError: If the flow name is not registered.

        Example:
            ```python
            app = ReMeApp()
            result = app.execute(
                "tool_memory_flow",
                query="How to use the search tool effectively?"
            )
            print(result)
            ```

        Note:
            For better performance in async contexts, prefer using async_execute() directly.
            This method creates a new event loop for each call, which has overhead.
        """
        return asyncio.run(self.async_execute(name=name, **kwargs))


def main():
    """
    Entry point for running ReMeApp as a service.

    This function initializes ReMeApp with command-line arguments and starts the service.
    It's typically called when running the module directly (python -m reme_ai.app).

    Command-line arguments are passed directly to ReMeApp.__init__(), allowing
    configuration via command line:

    Example:
        python -m reme_ai.app --llm_api_key=sk-xxx --config_path=config.yaml

    The app runs as a context manager, ensuring proper cleanup of resources
    (database connections, API clients, etc.) on shutdown.

    Note:
        Press Ctrl+C to gracefully shutdown the service.
    """
    with ReMeApp(*sys.argv[1:]) as app:
        app.run_service()


if __name__ == "__main__":
    main()
