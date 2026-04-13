"""Application configuration schema module.

This module defines the configuration models for the ReMe CLI application,
including application-level settings, service configuration, and job definitions.
"""

import os

from pydantic import BaseModel, ConfigDict, Field

from ..enumeration import ComponentEnum


class ComponentConfig(BaseModel):
    """Base configuration for a component.

    This serves as the base class for all component configurations,
    allowing extra fields to be defined dynamically.

    Attributes:
        backend: The backend implementation class name for this component.
    """

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="", description="Backend implementation class name")


class JobConfig(ComponentConfig):
    """Configuration for a job definition.

    A job represents a sequence of steps that can be executed
    as part of the application workflow.

    Attributes:
        name: Unique identifier name for the job.
        description: Human-readable description of what the job does.
        parameters: Job-level parameters passed to all steps.
        steps: Ordered list of step configurations to execute.
    """

    name: str = Field(default="", description="Unique job identifier")
    description: str = Field(default="", description="Human-readable job description")
    parameters: dict = Field(default_factory=dict, description="Job-level parameters")
    steps: list[ComponentConfig] = Field(default_factory=list, description="Ordered list of step configs")


class ApplicationConfig(BaseModel):
    """Root configuration for the ReMe CLI application.

    This model contains all configuration settings needed to initialize
    and run the application, including service endpoints, job definitions,
    and component registry.

    Attributes:
        app_name: Display name of the application.
        working_dir: Working directory for runtime files and logs.
        enable_logo: Whether to display the ASCII logo on startup.
        language: Default language for LLM interactions.
        log_to_console: Whether to output logs to console.
        log_to_file: Whether to write logs to a file.
        mcp_servers: MCP server configurations indexed by name.
        service: Service endpoint configuration.
        jobs: List of job definitions.
        components: Component registry indexed by component type and name.
    """

    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"), description="Application display name")
    working_dir: str = Field(default=".reme", description="Working directory for runtime files")
    enable_logo: bool = Field(default=False, description="Whether to show ASCII logo on startup")
    language: str = Field(default="", description="Default language for LLM interactions")
    log_to_console: bool = Field(default=True, description="Whether to log to console")
    log_to_file: bool = Field(default=True, description="Whether to log to file")
    mcp_servers: dict[str, dict] = Field(default_factory=dict, description="MCP server configurations")
    service: ComponentConfig = Field(default_factory=ComponentConfig, description="Service endpoint config")
    jobs: list[JobConfig] = Field(default_factory=list, description="Job definitions")
    components: dict[ComponentEnum, dict[str, ComponentConfig]] = Field(
        default_factory=dict, description="Component registry by type"
    )
