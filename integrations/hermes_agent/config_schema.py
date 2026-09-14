"""ReMe's configuration surface for Hermes' generic memory settings UI."""

# pylint: disable=no-name-in-module

from plugins.memory.config_schema import (
    KIND_NUMBER,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="reme",
    label="ReMe",
    docs_url="https://github.com/agentscope-ai/ReMe/tree/main/integrations/hermes_agent",
    fields=(
        ProviderField(
            key="mode",
            label="Mode",
            kind=KIND_SELECT,
            default="http",
            description="Choose a ReMe service or an in-process ReMe SDK.",
            options=(
                ProviderFieldOption(
                    "http",
                    "HTTP",
                    "Connect to an independently managed ReMe service",
                ),
                ProviderFieldOption(
                    "embedded",
                    "Embedded",
                    "Run ReMe inside the Hermes Python process",
                ),
            ),
            inline=True,
        ),
        ProviderField(
            key="endpoint",
            label="HTTP endpoint",
            kind=KIND_TEXT,
            default="http://127.0.0.1:2333",
            description="Used only in HTTP mode.",
            placeholder="http://127.0.0.1:2333",
            inline=True,
        ),
        ProviderField(
            key="workspace_dir",
            label="Workspace directory",
            kind=KIND_TEXT,
            default="",
            description="Required only in embedded mode. Use a separate workspace for each Hermes profile.",
            placeholder="~/.reme-hermes-default",
            inline=True,
        ),
        ProviderField(
            key="reme_config",
            label="ReMe configuration",
            kind=KIND_TEXT,
            default="default",
            description="Built-in config name or a YAML/JSON path used by embedded mode.",
            group="Embedded",
        ),
        ProviderField(
            key="recall_limit",
            label="Recall limit",
            kind=KIND_NUMBER,
            default="5",
            description="Maximum number of search results included before a model call.",
            group="Recall",
        ),
        ProviderField(
            key="recall_timeout",
            label="Recall timeout (seconds)",
            kind=KIND_NUMBER,
            default="5",
            group="Timeouts",
        ),
        ProviderField(
            key="request_timeout",
            label="Write/start timeout (seconds)",
            kind=KIND_NUMBER,
            default="600",
            group="Timeouts",
        ),
        ProviderField(
            key="health_timeout",
            label="Health timeout (seconds)",
            kind=KIND_NUMBER,
            default="2",
            group="Timeouts",
        ),
        ProviderField(
            key="health_retry_seconds",
            label="Health retry delay (seconds)",
            kind=KIND_NUMBER,
            default="30",
            group="Timeouts",
        ),
        ProviderField(
            key="shutdown_timeout",
            label="Shutdown timeout (seconds)",
            kind=KIND_NUMBER,
            default="30",
            group="Timeouts",
        ),
    ),
)
