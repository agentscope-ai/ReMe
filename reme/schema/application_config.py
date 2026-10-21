"""Application configuration models."""

from collections.abc import Mapping
from itertools import combinations
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..enumeration import component_type_name

DEFAULT_WORKSPACE_ALIAS = "default"


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
    enabled: bool = Field(default=True, description="Whether to start, serve, and allow execution of this job")


class ApplicationConfig(BaseModel):
    """Root config for the ReMe application."""

    app_name: str = Field(default=os.getenv("APP_NAME", "ReMe"), description="Application display name")
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables loaded once at startup and passed to agent subprocesses",
    )
    workspace_dir: str = Field(
        default=".reme",
        description=(
            "Workspace root directory for runtime files. Accepts a single path or an ordered mapping of "
            "alias to path; the mapping form normalizes to the first entry here and exposes all entries "
            "through the derived workspaces mapping."
        ),
        validate_default=True,
    )
    workspaces: dict[str, str] = Field(
        default_factory=dict,
        exclude=True,
        description="Ordered alias-to-absolute-path workspaces derived from workspace_dir",
    )
    metadata_dir: str = Field(default="metadata", description="Subdirectory for ReMe persistent state")
    session_dir: str = Field(default="session", description="Subdirectory for persisted agent sessions")
    # dialog_dir was removed; standard transcripts are always derived as ``{session_dir}/dialog``.
    mem_session_dir: str = Field(default="mem_session", description="Subdirectory for persisted agent sessions")
    resource_dir: str = Field(default="resource", description="Subdirectory for external assets")
    daily_dir: str = Field(default="daily", description="Subdirectory for daily memory")
    digest_dir: str = Field(default="digest", description="Subdirectory for digest memory")
    enable_logo: bool = Field(default=True, description="Show ASCII logo on startup")
    timezone: str | None = Field(default="Asia/Shanghai", description="IANA timezone; None uses local time")
    language: str = Field(default="", description="Default language for LLM interactions")
    log_to_console: bool = Field(default=True, description="Log to console")
    log_to_file: bool = Field(default=True, description="Log to file")
    mcp_servers: dict[str, dict] = Field(default_factory=dict, description="MCP server configs by name")
    plugins: list[str] = Field(default_factory=list, description="Installed plugins enabled for this app")
    service: ComponentConfig = Field(default_factory=ComponentConfig, description="Service endpoint config")
    jobs: dict[str, JobConfig] = Field(default_factory=dict, description="Job definitions keyed by job name")
    thread_pool_max_workers: int = Field(default=0, description="Max worker threads; 0 to disable")
    components: dict[str, dict[str, ComponentConfig]] = Field(
        default_factory=dict,
        description="Component registry keyed by type then name",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_workspace_mapping(cls, data):
        """Resolve a mapping workspace_dir, keeping its first entry as the default workspace."""
        if not isinstance(data, dict):
            return data
        if "workspaces" in data:
            raise ValueError("workspaces is derived from workspace_dir; do not supply it")
        raw = data.get("workspace_dir")
        if not isinstance(raw, Mapping):
            return data
        if not raw:
            raise ValueError("workspace_dir mapping must contain at least one workspace")
        resolved: dict[str, str] = {}
        for key, path in raw.items():
            if not isinstance(key, str):
                raise ValueError(f"workspace_dir aliases must be strings: {key!r}")
            alias = key.strip()
            if not alias:
                raise ValueError("workspace_dir aliases must be non-empty")
            if alias in resolved:
                raise ValueError(f"workspace_dir aliases collide after normalization: {alias!r}")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"workspace_dir[{key!r}] must be a non-empty path string: {path!r}")
            resolved[alias] = str(Path(path).expanduser().resolve(strict=False))
        directories = list(resolved.values())
        if len(set(directories)) != len(directories):
            raise ValueError(f"workspace_dir aliases must point to distinct directories: {resolved}")
        for first, second in combinations(map(Path, directories), 2):
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(
                    f"workspace_dir directories must not overlap: {str(first)!r} and {str(second)!r}",
                )
        return {**data, "workspace_dir": directories[0], "workspaces": resolved}

    @model_validator(mode="after")
    def derive_default_workspace(self):
        """Normalize every workspace_dir form to an ordered mapping; the first entry is the default."""
        self.workspaces = dict(self.workspaces) or {DEFAULT_WORKSPACE_ALIAS: self.workspace_dir}
        return self

    @field_validator("components", mode="before")
    @classmethod
    def normalize_component_types(cls, value):
        """Canonicalize built-in enums and allow plugin-defined component type names."""
        if value is None:
            return {}
        if not isinstance(value, dict):
            return value
        return {component_type_name(component_type): group for component_type, group in value.items()}

    @field_validator("workspace_dir", mode="before")
    @classmethod
    def normalize_workspace_dir(cls, value) -> str:
        """Expand home-relative paths once so every component sees the same absolute workspace."""
        return str(Path(value).expanduser().resolve(strict=False))

    @field_validator("session_dir")
    @classmethod
    def validate_session_dir(cls, value: str) -> str:
        """Keep standard session storage inside the configured workspace."""
        path = value.strip()
        if PurePosixPath(path.replace("\\", "/")).is_absolute() or PureWindowsPath(path).anchor:
            raise ValueError("session_dir must be a workspace-relative path")
        return value
