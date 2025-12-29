from typing import Dict, List

from pydantic import BaseModel, Field, ConfigDict

from .tool_call import ToolCall


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    transport: str = Field(default="")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)


class HttpConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8001)
    timeout_keep_alive: int = Field(default=3600)
    limit_concurrency: int = Field(default=1000)


class CmdConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    flow: str = Field(default="")


class FlowConfig(ToolCall):
    model_config = ConfigDict(extra="allow")

    flow_content: str = Field(default="")
    stream: bool = Field(default=False)
    enable_cache: bool = Field(default=False)
    cache_path: str = Field(default="cache/flow")
    cache_expire_hours: float = Field(default=0.1)


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    model_name: str = Field(default="")


class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    model_name: str = Field(default="")


class VectorStoreConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    collection_name: str = Field(default="")


class TokenCounterConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="base")
    model_name: str = Field(default="")


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="")
    enable_logo: bool = Field(default=True)
    language: str = Field(default="")
    thread_pool_max_workers: int = Field(default=16)
    ray_max_workers: int = Field(default=-1)
    init_logger: bool = Field(default=True)
    disabled_flows: List[str] = Field(default_factory=list)
    enabled_flows: List[str] = Field(default_factory=list)
    external_mcp: Dict[str, dict] = Field(default_factory=dict, description="External MCP Server config")

    mcp: MCPConfig = Field(default_factory=MCPConfig)
    http: HttpConfig = Field(default_factory=HttpConfig)
    cmd: CmdConfig = Field(default_factory=CmdConfig)
    flow: Dict[str, FlowConfig] = Field(default_factory=dict)
    llm: Dict[str, LLMConfig] = Field(default_factory=dict)
    embedding_model: Dict[str, EmbeddingModelConfig] = Field(default_factory=dict)
    vector_store: Dict[str, VectorStoreConfig] = Field(default_factory=dict)
    token_counter: Dict[str, TokenCounterConfig] = Field(default_factory=dict)
