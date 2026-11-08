"""Shared PA test wiring without a metadata-state fallback or provider startup."""

from reme.components import CodexAppServer, PaRuntime

PA_CONFIG = {"start_time": "0900", "end_time": "2300", "interval_minutes": 5}


def bind_pa(context, agent):
    """Bind one concrete session owner to an isolated PA runtime."""
    context.components["agent_wrapper"] = {"default": agent}
    server = context.components.get("codex_app_server", {}).get("default")
    if server is None:
        candidate = agent.app_server
        server = candidate if isinstance(candidate, CodexAppServer) else CodexAppServer(app_context=context)
    agent.app_server = server
    context.components["codex_app_server"] = {"default": server}
    runtime = PaRuntime(app_context=context, default_proactive=PA_CONFIG)
    runtime.agent_wrapper = agent
    context.components["pa"] = {"default": runtime}
    return runtime
