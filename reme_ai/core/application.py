import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from loguru import logger

from .context import C
from .enumeration import ChunkEnum, RegistryEnum
from .flow import BaseFlow, ExpressionFlow
from .schema import (
    ServiceConfig,
    LLMConfig,
    EmbeddingModelConfig,
    VectorStoreConfig,
    TokenCounterConfig,
    StreamChunk,
    Response,
)
from .service import BaseService
from .utils import FastMcpClient, PydanticConfigParser, init_logger, print_logo


class Application:
    def __init__(
        self,
        *args,
        llm_api_key: str = None,
        llm_api_base: str = None,
        embedding_api_key: str = None,
        embedding_api_base: str = None,
        llm: dict | None = None,
        embedding_model: dict | None = None,
        vector_store: dict | None = None,
        token_counter: dict | None = None,
        service_config: ServiceConfig = None,
        parser: type[PydanticConfigParser] = None,
        config_path: str = None,
        **kwargs,
    ):
        """
        Initialize application with configuration.

        Args:
            *args: Additional arguments passed to parser. Examples:
                - "llm.default.model_name=qwen3-30b-a3b-thinking-2507"
                - "llm.default.backend=openai_compatible"
                - "llm.default.temperature=0.6"
                - "embedding_model.default.model_name=text-embedding-v4"
                - "embedding_model.default.backend=openai_compatible"
                - "embedding_model.default.dimensions=1024"
                - "vector_store.default.backend=memory"
                - "vector_store.default.embedding_model=default"
            llm_api_key: API key for LLM service
            llm_api_base: Base URL for LLM API
            embedding_api_key: API key for embedding service
            embedding_api_base: Base URL for embedding API
            llm: Dictionary of default LLM configurations.
            embedding_model: Dictionary of default embedding model configurations.
            vector_store: Dictionary of default vector store configurations.
            token_counter: Dictionary of default token counter configurations.
            service_config: Pre-configured ServiceConfig object
            parser: Custom configuration parser class
            config_path: Path to custom configuration YAML file. If provided, loads configuration from this file.
                Example: "path/to/my_config.yaml"
            **kwargs: Additional keyword arguments passed to parser. Same format as args but as kwargs.
        """

        self._update_env("REME_LLM_API_KEY", llm_api_key)
        self._update_env("REME_LLM_BASE_URL", llm_api_base)
        self._update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self._update_env("REME_EMBEDDING_BASE_URL", embedding_api_base)

        self.parser = parser(ServiceConfig)
        self.service_config: ServiceConfig = service_config
        if self.service_config is None:
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)
            if kwargs:
                input_args.extend([f"{k}={v}" for k, v in kwargs.items()])
            self.service_config = self.parser.parse_args(*input_args)

        self._update_config_section("llm", llm)
        self._update_config_section("embedding_model", embedding_model)
        self._update_config_section("vector_store", vector_store)
        self._update_config_section("token_counter", token_counter)

        init_logger()

    @staticmethod
    def _update_env(key: str, value: str):
        if value:
            os.environ[key] = value

    def _update_config_section(self, section_name: str, update_dict: dict | None):
        if not update_dict:
            return

        target_registry = getattr(self.service_config, section_name)
        if "default" not in target_registry:
            raise KeyError(f"Default `{section_name}` config not found in service_config")

        current_config = target_registry["default"]
        target_registry["default"] = current_config.model_copy(update=update_dict, deep=True)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type=None, exc_val=None, exc_tb=None):
        await self.stop()
        return False

    async def _prepare_external_mcp(self):
        if not self.service_config.external_mcp:
            return

        for name, mcp_server_config in self.service_config.external_mcp.items():
            try:
                async with FastMcpClient(name=name, config=mcp_server_config) as client:
                    tool_calls = await client.list_tool_calls()
                    C.external_mcp_tool_call_dict[name] = {tool_call.name: tool_call for tool_call in tool_calls}
                    for tool_call in tool_calls:
                        logger.info(f"find_mcp: {name}@{tool_call.name} {tool_call.simple_input_dump()}")

            except Exception as e:
                logger.exception(f"get mcp@{name} tool_calls error: {e}")

    def _filter_flows(self, name: str) -> bool:
        if self.service_config.enabled_flows:
            return name in self.service_config.enabled_flows
        elif self.service_config.disabled_flows:
            return name not in self.service_config.disabled_flows
        else:
            return True

    def _update_service_context(self):
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
                **embedding_model_config.model_extra,
            )
            C.vector_store_dict[name] = vector_store_cls(
                collection_name=config.collection_name,
                embedding_model=embedding_model,
                **config.model_extra,
            )

        for name, flow_cls in C.registry_dict[RegistryEnum.FLOW].items():
            if not self._filter_flows(name):
                continue

            flow: BaseFlow = flow_cls()
            C.flow_dict[flow.name] = flow

        for name, flow_config in self.service_config.flow.items():
            if not self._filter_flows(name):
                continue

            flow_config.name = name
            flow: BaseFlow = ExpressionFlow(flow_config=flow_config)
            C.flow_dict[name] = flow

    def start_sync(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.start())
        else:
            import nest_asyncio

            nest_asyncio.apply()
            asyncio.run(self.start())

    async def stop(self, wait_thread_pool: bool = True, wait_ray: bool = True):
        for _, vector_store in C.vector_store_dict.items():
            await vector_store.close()
        C.thread_pool.shutdown(wait=wait_thread_pool)
        if self.service_config.ray_max_workers > 1:
            import ray

            ray.shutdown(_exiting_interpreter=not wait_ray)

    def stop_sync(self, wait_thread_pool: bool = True, wait_ray: bool = True):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.stop(wait_thread_pool, wait_ray))
        else:
            import nest_asyncio

            nest_asyncio.apply()
            asyncio.run(self.stop(wait_thread_pool, wait_ray))

    @staticmethod
    async def execute_flow(name: str, **kwargs) -> Response:
        flow: BaseFlow = C.get_flow(name)
        return await flow.call(**kwargs)

    @staticmethod
    def execute_flow_sync(name: str, **kwargs):
        flow: BaseFlow = C.get_flow(name)
        return flow.call_sync(**kwargs)

    @staticmethod
    async def execute_stream_flow(name: str, **kwargs):
        flow: BaseFlow = C.get_flow(name)
        assert flow.stream is True, "non-stream is not supported in execute_stream_flow!"
        stream_queue = asyncio.Queue()
        task = asyncio.create_task(flow.call(stream_queue=stream_queue, **kwargs))
        while True:
            try:
                stream_chunk: StreamChunk = await asyncio.wait_for(stream_queue.get(), timeout=1.0)
                if stream_chunk.done:
                    yield "data:[DONE]\n\n"
                    await task
                    break

                yield f"data:{stream_chunk.model_dump_json()}\n\n"

            except asyncio.TimeoutError:
                # Timeout: check if task has completed or failed
                if task.done():
                    try:
                        await task

                    except Exception as e:
                        logger.exception(f"flow={name} encounter error with args={e.args}")

                        error_chunk = StreamChunk(chunk_type=ChunkEnum.ERROR, chunk=str(e), done=True)
                        yield f"data:{error_chunk.model_dump_json()}\n\n"
                        yield "data:[DONE]\n\n"
                        break

                    else:
                        yield "data:[DONE]\n\n"
                        break

                continue

    def run_service(self):
        if self.service_config.enable_logo:
            print_logo(service_config=self.service_config, app_name=os.getenv("APP_NAME", "ReMe"))

        service_cls = C.get_service_class(self.service_config.backend)
        service: BaseService = service_cls(service_config=self.service_config)
        service.run()


# old service: sync __enter__ & __exit__ & run_service
# old import: async __aenter__ & async __aexit__ & execute_flow
# new import: __init__ & execute_flow & async_execute_flow
