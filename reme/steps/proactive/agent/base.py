"""Typed runtime access and workspace authorization shared by agent steps."""

from ...base_step import BaseStep, Ref
from ...file_io import check_path_permission
from ....components.agent_runtime import BaseAgentRuntime
from ....enumeration import ComponentEnum


class BaseAgentStep(BaseStep):
    """Keep request handlers stateless; the application component owns the runtime."""

    agent_runtime: BaseAgentRuntime = Ref(BaseAgentRuntime, ComponentEnum.AGENT_RUNTIME)

    @staticmethod
    def web_options(online: bool):
        """Apply one network policy to search, URL extraction and commands."""
        return {
            "job_tools": ["web_extract"] if online else [],
            "config": {
                "web_search": "live" if online else "disabled",
                "sandbox_workspace_write": {"network_access": online},
            },
        }

    def require_paths(self, *paths):
        """Preserve fail-closed tool bindings before accessing runtime files."""
        if any(
            not check_path_permission(self.workspace_path, path, self.context.get("_allowed_paths")) for path in paths
        ):
            raise PermissionError("no permission to access agent runtime files")

    def parameters(self):
        """Separate server-injected context from public request arguments."""
        return {
            key: value
            for key, value in self.context.data.items()
            if key not in {"metadata", "_allowed_paths", "_agent_id"}
        }
