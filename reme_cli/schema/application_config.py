"""Configuration schemas for service components using Pydantic models."""

import os

from pydantic import Field, BaseModel

from ..enumeration import ComponentEnum


class ApplicationConfig(BaseModel):
    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"))
    working_dir: str = Field(default=".reme")
    enable_logo: bool = Field(default=False)
    language: str = Field(default="")
    log_to_console: bool = Field(default=True)
    mcp_servers: dict[str, dict] = Field(default_factory=dict)
    service: dict = Field(default_factory=dict)
    ops: dict[str, dict] = Field(default_factory=dict)
    flows: dict[str, dict] = Field(default_factory=dict)
    components: dict[ComponentEnum, dict[str, dict]] = Field(default_factory=dict)
