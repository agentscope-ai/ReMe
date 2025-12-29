"""FlowLLM application core module for managing flows, services, and configurations."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .context import C
from .enumeration import RegistryEnum
from .flow import BaseFlow, ExpressionToolFlow
from .schema import ServiceConfig, EmbeddingModelConfig, FlowStreamChunk
from .service import BaseService
from .utils import FastMcpClient, PydanticConfigParser, init_logger, print_logo


class Application:
    """
    Base application class for FlowLLM framework.

    This is a generic, configurable base class that can be inherited to create
    custom application implementations. It manages the lifecycle of FlowLLM
    applications including configuration, service initialization, flow execution,
    and resource management. Supports both synchronous and asynchronous operation modes.

    This class can be used directly with custom parsers and configurations, or
    can be subclassed (e.g., FlowLLMApp) to provide default configurations
    and convenience methods for specific use cases.
    """

    def __init__(
        self,
        *args,
        llm_api_key: str = None,
        llm_api_base: str = None,
        embedding_api_key: str = None,
        embedding_api_base: str = None,
        service_config: ServiceConfig = None,
        parser: type[PydanticConfigParser] = None,
        config_path: str = None,
        load_default_config: bool = False,
        **kwargs,
    ):
        """
        Initialize FlowLLM application with configuration.

        Args:
            *args: Additional arguments passed to parser. Examples:
                - "llm.default.model_name=qwen3-30b-a3b-thinking-2507"
                - "llm.default.backend=openai_compatible"
                - "llm.default.params={'temperature': '0.6'}"
                - "embedding_model.default.model_name=text-embedding-v4"
                - "embedding_model.default.backend=openai_compatible"
                - "embedding_model.default.params={'dimensions': 1024}"
                - "vector_store.default.backend=memory"
                - "vector_store.default.embedding_model=default"
                - "vector_store.default.params={...}"
            llm_api_key: API key for LLM service
            llm_api_base: Base URL for LLM API
            embedding_api_key: API key for embedding service
            embedding_api_base: Base URL for embedding API
            service_config: Pre-configured ServiceConfig object
            parser: Custom configuration parser class
            config_path: Path to custom configuration YAML file. If provided, loads configuration from this file.
                Example: "path/to/my_config.yaml"
            load_default_config: Whether to load default configuration (default.yaml).
                If True and config_path is not provided, loads the default configuration.
            **kwargs: Additional keyword arguments passed to parser. Same format as args but as kwargs.
        """

        if llm_api_key:
            os.environ["FLOW_LLM_API_KEY"] = llm_api_key

        if llm_api_base:
            os.environ["FLOW_LLM_BASE_URL"] = llm_api_base

        if embedding_api_key:
            os.environ["FLOW_EMBEDDING_API_KEY"] = embedding_api_key

        if embedding_api_base:
            os.environ["FLOW_EMBEDDING_BASE_URL"] = embedding_api_base

        # Initialize parser first (needed for update_service_config method)
        if parser is None:
            parser = PydanticConfigParser
        self.parser = parser(ServiceConfig)

        if service_config is not None:
            self.service_config: ServiceConfig = service_config
        else:
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            elif load_default_config:
                input_args.append(f"config={parser.default_config_name}")

            if args:
                input_args.extend(args)

            if kwargs:
                input_args.extend([f"{k}={v}" for k, v in kwargs.items()])

            self.service_config = self.parser.parse_args(*input_args)

        if self.service_config.init_logger:
            init_logger()

    def update_service_config(self, **kwargs):
        """
        Update configuration object using keyword arguments

        Args:
            **kwargs: Configuration items to update, supports dot notation, e.g. a.b.c='xxx'

        Returns:
            Updated configuration object
        """
        self.service_config = self.parser.update_config(**kwargs)
        return self.service_config

    async def __aenter__(self):
        """
        Async context manager entry point.

        Returns:
            Self instance after starting the application
        """
        await self.async_start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit point.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        Returns:
            False to propagate any exception
        """
        await self.async_stop()
        return False

    @staticmethod
    async def get_mcp_tools(name: str, mcp_server_config: dict) -> dict:
        """
        Retrieve available tools from an MCP (Model Context Protocol) server.

        Args:
            name: Name identifier for the MCP server
            mcp_server_config: Configuration dictionary for the MCP server

        Returns:
            Dictionary containing server name and available tool calls, or empty dict on error
        """
        try:
            async with FastMcpClient(name=name, config=mcp_server_config) as client:
                tool_calls = await client.list_tool_calls()
                for tool_call in tool_calls:
                    logger.info(f"find mcp@{name}@{tool_call.name} {tool_call.model_dump_json()}")

                return {
                    "name": name,
                    "tool_calls": {tool_call.name: tool_call for tool_call in tool_calls},
                }

        except Exception as e:
            logger.exception(f"get mcp@{name} tool_calls error: {e}")
            return {}

    def filter_flows(self, name: str) -> bool:
        """
        Determine if a flow should be enabled based on configuration.

        Args:
            name: Flow name to check

        Returns:
            True if the flow should be enabled, False otherwise
        """
        if self.service_config.enabled_flows:
            return name in self.service_config.enabled_flows
        elif self.service_config.disabled_flows:
            return name not in self.service_config.disabled_flows
        else:
            return True

    async def async_start(self):
        """
        Asynchronously start the FlowLLM application.

        Initializes external MCP servers, service configuration, thread pools,
        vector stores, embedding models, and registers all flows.
        """
        # add external_mcp
        for name, mcp_server_config in self.service_config.external_mcp.items():
            mcp_server_info = await self.get_mcp_tools(name, mcp_server_config)
            if mcp_server_info:
                C.external_mcp_tool_call_dict[mcp_server_info["name"]] = mcp_server_info["tool_calls"]

        # add service_config & language & thread_pool & ray
        C.service_config = self.service_config
        C.language = self.service_config.language
        C.thread_pool = ThreadPoolExecutor(max_workers=self.service_config.thread_pool_max_workers)
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.init(num_cpus=self.service_config.ray_max_workers)

        # add vector store
        for name, config in self.service_config.vector_store.items():
            vector_store_cls = C.get_vector_store_class(config.backend)
            embedding_model_config: EmbeddingModelConfig = self.service_config.embedding_model[config.embedding_model]
            embedding_model_cls = C.get_embedding_model_class(embedding_model_config.backend)
            embedding_model = embedding_model_cls(
                model_name=embedding_model_config.model_name,
                **embedding_model_config.params,
            )
            C.vector_store_dict[name] = vector_store_cls(embedding_model=embedding_model, **config.params)

        # add cls flow
        for name, flow_cls in C.registry_dict[RegistryEnum.FLOW].items():
            if not self.filter_flows(name):
                continue

            flow: BaseFlow = flow_cls()
            C.flow_dict[flow.name] = flow

        # add expression flow
        for name, flow_config in self.service_config.flow.items():
            if not self.filter_flows(name):
                continue

            flow_config.name = name
            flow: BaseFlow = ExpressionToolFlow(flow_config=flow_config)
            C.flow_dict[name] = flow

    async def async_stop(self, wait_thread_pool=True, wait_ray: bool = True):
        """
        Asynchronously stop the FlowLLM application and clean up resources.

        Args:
            wait_thread_pool: Whether to wait for thread pool tasks to complete
            wait_ray: Whether to wait for Ray tasks to complete
        """
        for _, vector_store in C.vector_store_dict.items():
            await vector_store.async_close()
        C.thread_pool.shutdown(wait=wait_thread_pool)
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait_ray)

    def __enter__(self):
        """
        Synchronous context manager entry point.

        Returns:
            Self instance after starting the application
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Synchronous context manager exit point.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        Returns:
            False to propagate any exception
        """
        self.stop()
        return False

    def start(self):
        """
        Synchronously start the FlowLLM application.

        Wraps async_start() in asyncio.run() for synchronous usage.
        """
        asyncio.run(self.async_start())

    def stop(self, wait_thread_pool=True, wait_ray: bool = True):
        """
        Synchronously stop the FlowLLM application and clean up resources.

        Args:
            wait_thread_pool: Whether to wait for thread pool tasks to complete
            wait_ray: Whether to wait for Ray tasks to complete
        """
        for _, vector_store in C.vector_store_dict.items():
            vector_store.close()
        C.thread_pool.shutdown(wait=wait_thread_pool)
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait_ray)

    @staticmethod
    def execute_flow(name: str, **kwargs):
        """
        Synchronously execute a non-streaming flow by name.

        Args:
            name: Name of the flow to execute
            **kwargs: Additional arguments to pass to the flow

        Returns:
            Flow execution result

        Raises:
            AssertionError: If the flow is configured for streaming
        """
        flow: BaseFlow = C.get_flow(name)
        assert flow.stream is False, "stream is not supported in execute_flow!"
        return flow.call(**kwargs)

    @staticmethod
    async def async_execute_flow(name: str, **kwargs):
        """
        Asynchronously execute a non-streaming flow by name.

        Args:
            name: Name of the flow to execute
            **kwargs: Additional arguments to pass to the flow

        Returns:
            Flow execution result

        Raises:
            AssertionError: If the flow is configured for streaming
        """
        flow: BaseFlow = C.get_flow(name)
        assert flow.stream is False, "stream is not supported in async_execute_flow!"
        return await flow.async_call(**kwargs)

    @staticmethod
    async def async_execute_stream_flow(name: str, **kwargs):
        """
        Asynchronously execute a streaming flow by name.

        Args:
            name: Name of the flow to execute
            **kwargs: Additional arguments to pass to the flow

        Yields:
            Stream chunks in SSE (Server-Sent Events) format

        Raises:
            AssertionError: If the flow is not configured for streaming
        """
        flow: BaseFlow = C.get_flow(name)
        assert flow.stream is True, "non-stream is not supported in async_execute_stream_flow!"

        stream_queue = asyncio.Queue()
        asyncio.create_task(flow.async_call(stream_queue=stream_queue, **kwargs))
        while True:
            stream_chunk: FlowStreamChunk = await stream_queue.get()
            if stream_chunk.done:
                yield "data:[DONE]\n\n"
                break

            yield f"data:{stream_chunk.model_dump_json()}\n\n"

    def run_service(self):
        """
        Start the service based on the configured backend.

        Initializes and runs the appropriate service implementation
        (e.g., FastAPI, Flask) according to service configuration.
        """

        if self.service_config.enable_logo:
            print_logo(service_config=self.service_config, app_name=os.getenv("FLOW_APP_NAME"))

        service_cls = C.get_service_class(self.service_config.backend)
        service: BaseService = service_cls(service_config=self.service_config)
        service.run()
