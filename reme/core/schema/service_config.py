"""Configuration schemas for service components using Pydantic models."""

import os

from pydantic import BaseModel, Field, ConfigDict

from .tool_call import ToolCall


class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol transport and network settings."""

    model_config = ConfigDict(extra="allow")

    transport: str = Field(default="stdio")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)


class HttpConfig(BaseModel):
    """Configuration for the HTTP server interface and connection lifecycle."""

    model_config = ConfigDict(extra="allow")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    timeout_keep_alive: int = Field(default=3600)
    limit_concurrency: int = Field(default=1000)


class CmdConfig(BaseModel):
    """Configuration for command-line flow execution parameters."""

    model_config = ConfigDict(extra="allow")

    flow: str = Field(default="")


class OpConfig(BaseModel):
    """Configuration for op settings and parameters."""

    model_config = ConfigDict(extra="allow")

    prompt_dict: dict[str, str] = Field(default_factory=dict)
    params: dict = Field(default_factory=dict)


class FlowConfig(ToolCall):
    """Configuration for workflow execution, caching, and error handling."""

    model_config = ConfigDict(extra="allow")

    flow_content: str = Field(default="")
    stream: bool = Field(default=False)
    raise_exception: bool = Field(default=True)
    enable_cache: bool = Field(default=False)
    cache_path: str = Field(default="cache/flow")
    cache_expire_hours: float = Field(default=0.1)


class BasicConfig(BaseModel):
    """Configuration for basic service settings and parameters."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")


class ModelConfig(BasicConfig):
    """Configuration for model-based services with backend and model name."""

    model_name: str = Field(default="")


class LLMConfig(ModelConfig):
    """Configuration for Large Language Model backend and model identification."""


class EmbeddingModelConfig(ModelConfig):
    """Configuration for embedding model backends and identity."""


class TokenCounterConfig(ModelConfig):
    """Configuration for token counting services and model mapping."""


class StoreConfig(BasicConfig):
    """Configuration for storage services with embedding model support."""

    embedding_model: str = Field(default="default")


class VectorStoreConfig(StoreConfig):
    """Configuration for vector database storage and associated embeddings."""

    collection_name: str = Field(default="reme")


class FileStoreConfig(StoreConfig):
    """Configuration for file store database storage and associated embeddings."""

    store_name: str = Field(default="reme")


class FileWatcherConfig(BasicConfig):
    """Configuration for file watcher service."""

    file_store: str = Field(default="")
    watch_paths: list[str] = Field(default_factory=list)


class ServiceConfig(BasicConfig):
    """Root configuration schema aggregating all service-level settings and components."""

    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"))
    working_dir: str = Field(default=".reme")
    enable_logo: bool = Field(default=True)
    language: str = Field(default="")
    thread_pool_max_workers: int = Field(default=16)
    ray_max_workers: int = Field(default=-1)
    log_to_console: bool = Field(default=True)
    disabled_flows: list[str] = Field(default_factory=list)
    enabled_flows: list[str] = Field(default_factory=list)

    mcp_servers: dict[str, dict] = Field(default_factory=dict)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    cmd: CmdConfig = Field(default_factory=CmdConfig)
    ops: dict[str, OpConfig] = Field(default_factory=dict)
    flows: dict[str, FlowConfig] = Field(default_factory=dict)
    as_llms: dict[str, BasicConfig] = Field(default_factory=dict)
    as_llm_formatters: dict[str, BasicConfig] = Field(default_factory=dict)
    as_token_counters: dict[str, BasicConfig] = Field(default_factory=dict)
    llms: dict[str, LLMConfig] = Field(default_factory=dict)
    embedding_models: dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    vector_stores: dict[str, VectorStoreConfig] = Field(default_factory=dict)
    file_stores: dict[str, FileStoreConfig] = Field(default_factory=dict)
    token_counters: dict[str, TokenCounterConfig] = Field(default_factory=dict)
    file_watchers: dict[str, FileWatcherConfig] = Field(default_factory=dict)

    metadata: dict = Field(default_factory=dict)
