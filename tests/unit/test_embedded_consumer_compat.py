"""Compatibility tests for applications that embed ReMe in-process."""

import asyncio

from reme import ReMe
from reme.components.agent_wrapper import AsAgentWrapper
from reme.enumeration import ComponentEnum


def _qwenpaw_style_config(workspace_dir: str) -> dict:
    """Return the narrow ReMe contract used by QwenPaw's memory manager."""
    return {
        "workspace_dir": workspace_dir,
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "service": {"backend": "http"},
        "jobs": {
            "version": {
                "backend": "base",
                "description": "return reme package version",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "version_step"}],
            },
        },
        "components": {
            "as_llm": {
                "default": {
                    "backend": "openai",
                    "model": "consumer-injected",
                    "credential": {"api_key": "", "base_url": ""},
                },
            },
            "agent_wrapper": {
                "default": {
                    "backend": "agentscope",
                    "as_llm": "default",
                },
            },
        },
    }


def test_qwenpaw_style_config_preserves_optional_defaults(tmp_path):
    """New application fields remain optional for existing embedded configs."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))

    assert app.config.environment == {}
    assert app.context.service is not None
    assert app.context.service.jobs is None

    wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]
    assert isinstance(wrapper, AsAgentWrapper)
    assert wrapper.subprocess_environment == {}


def test_qwenpaw_style_config_keeps_in_process_application_api(tmp_path):
    """Model injection, lifecycle, and direct job execution remain compatible."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    injected_model = object()

    async def exercise_api() -> None:
        component = await app.update_component(
            "as_llm",
            "default",
            model=injected_model,
        )
        await app.start()
        try:
            response = await app.run_job("version")

            assert component.model is injected_model
            assert response.success is True
            assert response.answer
        finally:
            await app.close()

        assert app.is_started is False

    asyncio.run(exercise_api())
