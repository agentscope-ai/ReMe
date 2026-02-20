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


class FlowConfig(ToolCall):
    """Configuration for workflow execution, caching, and error handling."""

    model_config = ConfigDict(extra="allow")

    flow_content: str = Field(default="")
    stream: bool = Field(default=False)
    raise_exception: bool = Field(default=True)
    enable_cache: bool = Field(default=False)
    cache_path: str = Field(default="cache/flow")
    cache_expire_hours: float = Field(default=0.1)


class LLMConfig(BaseModel):
    """Configuration for Large Language Model backend and model identification."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    model_name: str = Field(default="")


class EmbeddingModelConfig(BaseModel):
    """Configuration for embedding model backends and identity."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    model_name: str = Field(default="")


class VectorStoreConfig(BaseModel):
    """Configuration for vector database storage and associated embeddings."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="local")
    collection_name: str = Field(default="reme")
    embedding_model: str = Field(default="default")


class MemoryStoreConfig(BaseModel):
    """Configuration for memory database storage and associated embeddings."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="sqlite")
    db_name: str = Field(default="reme.db")
    store_name: str = Field(default="reme")
    embedding_model: str = Field(default="default")


class TokenCounterConfig(BaseModel):
    """Configuration for token counting services and model mapping."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="base")
    model_name: str = Field(default="")


class FileWatcherConfig(BaseModel):
    """Configuration for file watcher service."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    memory_store: str = Field(default="")
    watch_paths: list[str] = Field(default_factory=list)


class ServiceConfig(BaseModel):
    """Root configuration schema aggregating all service-level settings and components."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"))
    working_dir: str | None = Field(default=None)
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
    flows: dict[str, FlowConfig] = Field(default_factory=dict)
    llms: dict[str, LLMConfig] = Field(default_factory=dict)
    embedding_models: dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    vector_stores: dict[str, VectorStoreConfig] = Field(default_factory=dict)
    memory_stores: dict[str, MemoryStoreConfig] = Field(default_factory=dict)
    token_counters: dict[str, TokenCounterConfig] = Field(default_factory=dict)
    file_watchers: dict[str, FileWatcherConfig] = Field(default_factory=dict)

    metadata: dict = Field(default_factory=dict)
