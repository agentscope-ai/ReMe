"""High-level entry point for configuring and running ReMe services and flows."""

import asyncio

from .context import PromptHandler, ServiceContext
from .embedding import BaseEmbeddingModel
from .file_watcher import BaseFileWatcher
from .flow import BaseFlow
from .llm import BaseLLM
from .memory_store import BaseMemoryStore
from .schema import Response, ServiceConfig
from .token_counter import BaseTokenCounter
from .utils import execute_stream_task, PydanticConfigParser
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
        default_memory_store_config: dict | None = None,
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
            default_memory_store_config=default_memory_store_config,
            default_token_counter_config=default_token_counter_config,
            default_file_watcher_config=default_file_watcher_config,
            **kwargs,
        )
        self.prompt_handler = PromptHandler(language=self.service_context.language)
        self._started: bool = False

    def update_api_envs(
        self,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
    ):
        """Update the API environment variables."""
        self.service_context.update_api_envs(
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
        )

    @classmethod
    async def create(
        cls,
        *args,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        enable_logo: bool = True,
        parser: type[PydanticConfigParser] | None = None,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        memory_store: dict | None = None,
        token_counter: dict | None = None,
        file_watcher: dict | None = None,
        **kwargs,
    ) -> "Application":
        """Create and start an Application instance asynchronously."""
        instance = cls(
            *args,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            embedding_api_key=embedding_api_key,
            embedding_base_url=embedding_base_url,
            enable_logo=enable_logo,
            parser=parser,
            default_llm_config=llm,
            default_embedding_model_config=embedding_model,
            default_vector_store_config=vector_store,
            default_memory_store_config=memory_store,
            default_token_counter_config=token_counter,
            default_file_watcher_config=file_watcher,
            **kwargs,
        )
        await instance.start()
        return instance

    async def start(self):
        """Start the application."""
        if self._started:
            return self
        else:
            await self.service_context.start()
            self._started = True
            return self

    async def close(self):
        """Close the application."""
        if self._started:
            await self.service_context.close()
            self._started = False
        else:
            raise RuntimeError("Application is not started")
        return False

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
    def default_memory_store(self) -> BaseMemoryStore:
        """Get the default memory store instance."""
        return self.service_context.memory_stores.get("default")

    def get_memory_store(self, name: str):
        """Get a memory store instance by name."""
        return self.service_context.memory_stores.get(name)

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

    @property
    def service_config(self) -> ServiceConfig:
        """Get the service configuration."""
        return self.service_context.service_config

    def get_token_counter(self, name: str):
        """Get a token counter instance by name."""
        return self.service_context.token_counters.get(name)

    def run_service(self):
        """Run the configured service (HTTP, MCP, or CMD)."""
        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.service_context.service.run()
