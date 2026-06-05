"""Application configuration models."""

import os

from pydantic import BaseModel, ConfigDict, Field

from ..enumeration import ComponentEnum


class ComponentConfig(BaseModel):
    """Base config for a component; extra fields allowed for backend-specific options."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="", description="Backend implementation class name")


class JobConfig(ComponentConfig):
    """Config for a job — an ordered sequence of step components. Keyed by name in ApplicationConfig.jobs."""

    description: str = Field(default="", description="Human-readable description")
    parameters: dict = Field(default_factory=dict, description="Job-level parameters")
    steps: list[ComponentConfig] = Field(default_factory=list, description="Ordered step configs")
    enable_serve: bool = Field(default=True, description="Whether to expose this job through the service layer")


class ApplicationConfig(BaseModel):
    """Root config for the ReMe application."""

    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"), description="Application display name")
    vault_dir: str = Field(default=".reme", description="Vault root directory for runtime files")
    metadata_dir: str = Field(default="reme_metadata", description="Subdirectory for ReMe persistent state")
    resource_dir: str = Field(default="resource", description="Subdirectory for external assets")
    daily_dir: str = Field(default="daily", description="Subdirectory for daily memory")
    digest_dir: str = Field(default="digest", description="Subdirectory for digest memory")
    enable_logo: bool = Field(default=True, description="Show ASCII logo on startup")
    timezone: str | None = Field(default=None, description="IANA timezone (e.g. 'Asia/Shanghai'); None uses local time")
    language: str = Field(default="", description="Default language for LLM interactions")
    log_to_console: bool = Field(default=True, description="Log to console")
    log_to_file: bool = Field(default=True, description="Log to file")
    mcp_servers: dict[str, dict] = Field(default_factory=dict, description="MCP server configs by name")
    service: ComponentConfig = Field(default_factory=ComponentConfig, description="Service endpoint config")
    jobs: dict[str, JobConfig] = Field(default_factory=dict, description="Job definitions keyed by job name")
    thread_pool_max_workers: int = Field(default=0, description="Max worker threads; 0 to disable")
    components: dict[ComponentEnum, dict[str, ComponentConfig]] = Field(
        default_factory=dict,
        description="Component registry keyed by type then name",
    )
