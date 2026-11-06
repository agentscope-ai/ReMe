"""Shared-server SDK doubles for workspace ownership and tool-boundary tests."""

# pylint: disable=protected-access

from reme.components.codex_app_server import CodexAppServer
from reme.enumeration import ComponentEnum

from .test_codex_agent_wrapper import _Codex, _Thread


class WorkspaceCodex(_Codex):
    """Permit first use of an external thread while retaining real server lifecycle hooks."""

    async def thread_resume(self, thread_id, **options):
        if thread_id not in self.threads:
            self.threads[thread_id] = _Thread(self, thread_id)
        return await super().thread_resume(thread_id, **options)


def workspace_server(context, monkeypatch):
    """Register the application-owned server without contacting a provider."""
    monkeypatch.setattr("openai_codex.AsyncCodex", WorkspaceCodex)
    server = CodexAppServer(name="default", app_context=context, auth_mode="oauth")
    context.shared_components.setdefault(ComponentEnum.CODEX_APP_SERVER, {})["default"] = server
    return server
