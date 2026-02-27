"""Service context."""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from loguru import logger

from .base_dict import BaseDict
from .schema import ServiceConfig
from .utils import load_env, PydanticConfigParser

if TYPE_CHECKING:
    from .llm import BaseLLM
    from .embedding import BaseEmbeddingModel
    from .vector_store import BaseVectorStore
    from .file_store import BaseFileStore
    from .token_counter import BaseTokenCounter
    from .flow import BaseFlow
    from .file_watcher import BaseFileWatcher


class ServiceContext(BaseDict):
    """Service context."""

    def __init__(
        self,
        *args,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        service_config: ServiceConfig | None = None,
        parser: type[PydanticConfigParser] | None = None,
        working_dir: str | None = None,
        config_path: str | None = None,
        enable_logo: bool = True,
        log_to_console: bool = True,
        default_llm_config: dict | None = None,
        default_embedding_model_config: dict | None = None,
        default_vector_store_config: dict | None = None,
        default_file_store_config: dict | None = None,
        default_token_counter_config: dict | None = None,
        default_file_watcher_config: dict | None = None,
        **kwargs,
    ):
        super().__init__()

        # Load environment variables
        load_env()

        # Update common environment variables for LLM and embedding services.
        self.update_env("REME_LLM_API_KEY", llm_api_key)
        self.update_env("REME_LLM_BASE_URL", llm_base_url)
        self.update_env("REME_EMBEDDING_API_KEY", embedding_api_key)
        self.update_env("REME_EMBEDDING_BASE_URL", embedding_base_url)

        if service_config is None:
            parser_class = parser if parser is not None else PydanticConfigParser
            parser_instance = parser_class(ServiceConfig)
            input_args = []
            if config_path:
                input_args.append(f"config={config_path}")
            if args:
                input_args.extend(args)

            if default_llm_config:
                self._update_section_config(kwargs, "llms", **default_llm_config)
            if default_embedding_model_config:
                self._update_section_config(kwargs, "embedding_models", **default_embedding_model_config)
            if default_token_counter_config:
                self._update_section_config(kwargs, "token_counters", **default_token_counter_config)
            if default_vector_store_config:
                self._update_section_config(kwargs, "vector_stores", **default_vector_store_config)
            if default_file_store_config:
                self._update_section_config(kwargs, "file_stores", **default_file_store_config)
            if default_file_watcher_config:
                self._update_section_config(kwargs, "file_watchers", **default_file_watcher_config)

            kwargs.update(
                {
                    "enable_logo": enable_logo,
                    "log_to_console": log_to_console,
                    "working_dir": working_dir,
                },
            )
            logger.info(f"update with args: {input_args} kwargs: {kwargs}")
            service_config = parser_instance.parse_args(*input_args, **kwargs)

        self.service_config: ServiceConfig = service_config

        self.thread_pool: ThreadPoolExecutor | None = None
        self.llms: dict[str, "BaseLLM"] = {}
        self.embedding_models: dict[str, "BaseEmbeddingModel"] = {}
        self.token_counters: dict[str, "BaseTokenCounter"] = {}
        self.vector_stores: dict[str, "BaseVectorStore"] = {}
        self.file_stores: dict[str, "BaseFileStore"] = {}
        self.file_watchers: dict[str, "BaseFileWatcher"] = {}
        self.flows: dict[str, "BaseFlow"] = {}
        self.mcp_server_mapping: dict[str, dict] = {}

    @staticmethod
    def update_env(key: str, value: str | None):
        """Update environment variable if value is provided."""
        if value:
            os.environ[key] = value

    @staticmethod
    def _update_section_config(config: dict, section_name: str, **kwargs):
        """Update a specific section of the service config with new values."""
        if section_name not in config:
            config[section_name] = {}
        if "default" not in config[section_name]:
            config[section_name]["default"] = {}
        config[section_name]["default"].update(kwargs)
