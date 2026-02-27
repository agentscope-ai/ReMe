"""High-level entry point for configuring and running ReMe services and flows."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from .embedding import BaseEmbeddingModel
from .file_store import BaseFileStore
from .file_watcher import BaseFileWatcher
from .flow import BaseFlow
from .llm import BaseLLM
from .prompt_handler import PromptHandler
from .registry_factory import R
from .schema import Response, ServiceConfig
from .service_context import ServiceContext
from .token_counter import BaseTokenCounter
from .utils import execute_stream_task, PydanticConfigParser, init_logger, MCPClient, print_logo
from .vector_store import BaseVectorStore


class Application:
    """Application wrapper that wires together service context, flows, and runtimes."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        working_dir: str | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        log_to_console: bool = True,
        parser: type[PydanticConfigParser] | None = None,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_vector_store_config: dict | None = None,
        default_file_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        **kwargs,
    ):
        self.service_context = ServiceContext(
            *args,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            service_config=None,
            parser=parser,
            working_dir=working_dir,
            config_path=config_path,
            enable_logo=enable_logo,
            log_to_console=log_to_console,
            default_llm_config=default_llm_config,
            default_embedding_model_config=default_embedding_model_config,
            default_vector_store_config=default_vector_store_config,
            default_file_store_config=default_file_store_config,
            default_token_counter_config=default_token_counter_config,
            default_file_watcher_config=default_file_watcher_config,
            **kwargs,
        )

        self.prompt_handler = PromptHandler(language=self.service_config.language)

        # NOTE: flows are initialized here to start service!
        self.init_flows()

        self._started: bool = False

    @classmethod
    async def create(cls, *args, **kwargs) -> "Application":
        """Create and start an Application instance asynchronously."""
        instance = cls(*args, **kwargs)
        await instance.start()
        return instance

    def init_flows(self):
        """Initialize flows."""
        expression_flow_cls = None
        for name, flow_cls in R.flows.items():
            if not self._filter_flows(name):
                continue

            if name == "ExpressionFlow":
                expression_flow_cls = flow_cls
            else:
                flow: "BaseFlow" = flow_cls(name=name, service_context=self.service_context)
                self.service_context.flows[flow.name] = flow

        if expression_flow_cls is not None:
            for name, flow_config in self.service_config.flows.items():
                if not self._filter_flows(name):
                    continue
                flow_config.name = name
                flow: BaseFlow = expression_flow_cls(  # noqa
                    flow_config=flow_config,
                    service_context=self.service_context,
                )
                self.service_context.flows[flow.name] = flow
        else:
            logger.info("No expression flow found, please check your configuration.")

    def _filter_flows(self, name: str) -> bool:
        """Filter flows based on enabled_flows and disabled_flows configuration."""
        if self.service_config.enabled_flows:
            return name in self.service_config.enabled_flows
        elif self.service_config.disabled_flows:
            return name not in self.service_config.disabled_flows
        else:
            return True

    @property
    def service_config(self) -> ServiceConfig:
        """Get the service configuration."""
        return self.service_context.service_config

    async def start(self):
        """Start the service context by initializing all configured components."""
        if self._started:
            logger.warning("Application has already started.")
            return self

        init_logger(log_to_console=self.service_config.log_to_console)
        logger.info(f"Init ReMe with config: {self.service_config.model_dump_json()}")

        working_path = Path(self.service_config.working_dir)
        working_path.mkdir(parents=True, exist_ok=True)

        if self.service_config.ray_max_workers > 1:
            import ray

            if not ray.is_initialized():
                ray.init(num_cpus=self.service_config.ray_max_workers)

        if (
            self.service_context.thread_pool is None
            or self.service_context.thread_pool._shutdown  # pylint: disable=protected-access
        ):
            self.service_context.thread_pool = ThreadPoolExecutor(
                max_workers=self.service_config.thread_pool_max_workers,
            )

        if self.service_context.service_config.enable_logo:
            print_logo(service_config=self.service_config)

        for name, config in self.service_config.llms.items():
            if config.backend not in R.llms:
                logger.warning(f"LLM backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend"})
                self.service_context.llms[name] = R.llms[config.backend](**config_dict)
                await self.service_context.llms[name].start()

        for name, config in self.service_config.embedding_models.items():
            if config.backend not in R.embedding_models:
                logger.warning(f"Embedding model backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend"})
                config_dict["cache_dir"] = working_path / "embedding_cache"
                self.service_context.embedding_models[name] = R.embedding_models[config.backend](**config_dict)
                await self.service_context.embedding_models[name].start()

        for name, config in self.service_config.token_counters.items():
            if config.backend not in R.token_counters:
                logger.warning(f"Token counter backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend"})
                self.service_context.token_counters[name] = R.token_counters[config.backend](**config_dict)

        for name, config in self.service_config.vector_stores.items():
            if config.backend not in R.vector_stores:
                logger.warning(f"Vector store backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend", "embedding_model"})
                config_dict.update(
                    {
                        "embedding_model": self.service_context.embedding_models[config.embedding_model],
                        "db_path": working_path / "vector_store",
                    },
                )
                self.service_context.vector_stores[name] = R.vector_stores[config.backend](**config_dict)
                await self.service_context.vector_stores[name].start()

        for name, config in self.service_config.file_stores.items():
            if config.backend not in R.file_stores:
                logger.warning(f"File store backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend", "embedding_model"})
                config_dict.update(
                    {
                        "embedding_model": self.service_context.embedding_models[config.embedding_model],
                        "db_path": working_path / "file_store",
                    },
                )
                self.service_context.file_stores[name] = R.file_stores[config.backend](**config_dict)
                await self.service_context.file_stores[name].start()

        for name, config in self.service_config.file_watchers.items():
            if config.backend not in R.file_watchers:
                logger.warning(f"File watcher backend {config.backend} is not supported.")
            else:
                config_dict = config.model_dump(exclude={"backend", "file_store"})
                config_dict["file_store"] = self.service_context.file_stores[config.file_store]
                self.service_context.file_watchers[name] = R.file_watchers[config.backend](**config_dict)
                await self.service_context.file_watchers[name].start()

        if self.service_config.mcp_servers:
            await self.prepare_mcp_servers()

        self._started = True
        return self

    async def prepare_mcp_servers(self):
        """Prepare and initialize MCP server connections."""
        mcp_client = MCPClient(config={"mcpServers": self.service_config.mcp_servers})
        for server_name in self.service_config.mcp_servers.keys():
            try:
                tool_calls = await mcp_client.list_tool_calls(server_name=server_name, return_dict=False)
                self.service_context.mcp_server_mapping[server_name] = {
                    tool_call.name: tool_call for tool_call in tool_calls
                }
                for tool_call in tool_calls:
                    logger.info(f"list_tool_calls: {server_name}@{tool_call.name} {tool_call.simple_input_dump()}")
            except Exception as e:
                logger.exception(f"list_tool_calls: {server_name} error: {e}")

    async def close(self) -> bool:
        """Close all service components asynchronously."""
        if not self._started:
            logger.warning("Application is not started")
            return True

        for name, vector_store in self.service_context.vector_stores.items():
            logger.info(f"Closing vector store: {name}")
            await vector_store.close()

        for name, file_store in self.service_context.file_stores.items():
            logger.info(f"Closing file store: {name}")
            await file_store.close()

        for name, file_watcher in self.service_context.file_watchers.items():
            logger.info(f"Closing file watcher: {name}")
            await file_watcher.close()

        for name, llm in self.service_context.llms.items():
            logger.info(f"Closing LLM: {name}")
            await llm.close()

        for name, embedding_model in self.service_context.embedding_models.items():
            logger.info(f"Closing embedding model: {name}")
            await embedding_model.close()

        self.shutdown_thread_pool()
        self.shutdown_ray()

        self._started = False
        return False

    def shutdown_thread_pool(self, wait: bool = True):
        """Shutdown the thread pool executor."""
        if self.service_context.thread_pool:
            self.service_context.thread_pool.shutdown(wait=wait)

    def shutdown_ray(self, wait: bool = True):
        """Shutdown Ray cluster if it was initialized."""
        if self.service_config and self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait)

    async def __aenter__(self):
        """Async context manager entry."""
        return await self.start()

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        """Async context manager exit."""
        return await self.close()

    async def execute_flow(self, name: str, **kwargs) -> Response:
        """Execute a flow with the given name and parameters."""
        assert name in self.service_context.flows, f"Flow {name} not found"
        flow: BaseFlow = self.service_context.flows[name]
        return await flow.call(**kwargs)

    async def execute_stream_flow(self, name: str, **kwargs):
        """Execute a stream flow with the given name and parameters."""
        assert name in self.service_context.flows, f"Flow {name} not found"
        flow: BaseFlow = self.service_context.flows[name]
        assert flow.stream is True, "non-stream flow is not supported in execute_stream_flow!"
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(flow.call(stream_queue=stream_queue, **kwargs))
        async for chunk in execute_stream_task(
            stream_queue=stream_queue,
            task=task,
            task_name=name,
            output_format="str",
        ):
            yield chunk

    @property
    def default_llm(self) -> BaseLLM:
        """Get the default LLM instance."""
        return self.service_context.llms.get("default")

    def get_llm(self, name: str):
        """Get an LLM instance by name."""
        return self.service_context.llms.get(name)

    def update_default_llm_name(self, name: str):
        """Update the default LLM name."""
        self.default_llm.model_name = name

    @property
    def default_embedding_model(self) -> BaseEmbeddingModel:
        """Get the default embedding model instance."""
        return self.service_context.embedding_models.get("default")

    def get_embedding_model(self, name: str):
        """Get an embedding model instance by name."""
        return self.service_context.embedding_models.get(name)

    def update_default_embedding_name(self, name: str):
        """Update the default embedding model name."""
        self.default_embedding_model.model_name = name

    @property
    def default_vector_store(self) -> BaseVectorStore:
        """Get the default vector store instance."""
        return self.service_context.vector_stores.get("default")

    def get_vector_store(self, name: str):
        """Get a vector store instance by name."""
        return self.service_context.vector_stores.get(name)

    @property
    def default_file_store(self) -> BaseFileStore:
        """Get the default file store instance."""
        return self.service_context.file_stores.get("default")

    def get_file_store(self, name: str):
        """Get a file store instance by name."""
        return self.service_context.file_stores.get(name)

    @property
    def default_file_watcher(self) -> BaseFileWatcher:
        """Get the default file watcher instance."""
        return self.service_context.file_watchers.get("default")

    def get_file_watcher(self, name: str):
        """Get a file watcher instance by name."""
        return self.service_context.file_watchers.get(name)

    @property
    def default_token_counter(self) -> BaseTokenCounter:
        """Get the default token counter instance."""
        return self.service_context.token_counters.get("default")

    def get_token_counter(self, name: str):
        """Get a token counter instance by name."""
        return self.service_context.token_counters.get(name)

    def run_service(self):
        """Run the configured service (HTTP, MCP, or CMD)."""
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        service = R.services[self.service_config.backend](app=self)
        service.run()

    async def reset_default_collection(self, collection_name: str):
        """Reset the default vector store."""
        await self.service_context.vector_stores["default"].reset_collection(collection_name)
