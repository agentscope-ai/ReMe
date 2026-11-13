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


def task(identifier="one", **overrides):
    """Build a current cron definition."""
    return {"id": identifier, "cron": "*/5 * * * *", "query": "观察屏幕", "enabled": True, **overrides}


def configure(env, definitions):
    """Publish user-owned task definitions."""
    env.runtime.files.write_json(env.runtime.cron_path, definitions)


def snapshot(env, identifier="one", time="120000", day="2026-11-08", status="unread", content="累计发现"):
    """Write a published result with independent consumption state."""
    import frontmatter

    path = env.runtime.tasks.task_path(identifier) / day / f"{time}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter.dumps(frontmatter.Post(content, task_id=identifier, status=status)))
    return path


def stub_model(env, monkeypatch, *, content="此前总结。\n\n本次发现。", no_update=False, message=None):
    """Mock the model boundary while exercising actual publication and delivery tools."""
    from unittest.mock import AsyncMock
    from reme.steps.proactive import CronAgentStep

    captured = {}
    original = CronAgentStep.prompt_format
    if not env.runtime.tasks.load():
        configure(env, [task()])

    def prompt(step, name, **kwargs):
        if name == "query":
            captured.update(kwargs)
        return original(step, name, **kwargs)

    monkeypatch.setattr(CronAgentStep, "prompt_format", prompt)

    async def reply(_query, **kwargs):
        binding = kwargs["injected_job_kwargs"]
        state = env.runtime.tasks.execution(binding["cron_task_id"]).state
        if not no_update:
            env.runtime.tasks.staging_path(state.run).write_text(content, encoding="utf-8")
        if message is not None:
            await message(binding)
        return {
            "session_id": kwargs["resume"],
            "status": "completed",
            "result": "NO_UPDATE" if no_update else "finished",
        }

    env.agent.reply = AsyncMock(side_effect=reply)
    return captured


async def wait_until(predicate):
    """Wait for an observable background state with a bounded deadline."""
    import asyncio

    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)
