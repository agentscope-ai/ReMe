"""Compatibility tests for applications that embed ReMe in-process."""

# pylint: disable=protected-access

import asyncio

import pytest

from reme import ReMe
from reme.components.agent_wrapper import AsAgentWrapper
from reme.components.as_llm import BaseAsLLM, DashScopeAsLLM
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


def test_update_component_validates_all_fields_before_mutation(tmp_path):
    """A rejected field update does not leave earlier attributes changed."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    component = app.context.components[ComponentEnum.AS_LLM]["default"]
    original_model = component.model

    async def exercise_api() -> None:
        with pytest.raises(AttributeError, match="does_not_exist"):
            await app.update_component(
                "as_llm",
                "default",
                model=object(),
                does_not_exist=True,
            )

        assert component.model is original_model

    asyncio.run(exercise_api())


def test_replace_component_rebinds_dependents_and_reuses_runtime_model(tmp_path):
    """A live backend replacement switches the wrapper and every bind target."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    old_model = object()
    verified_model = object()

    async def exercise_api() -> None:
        old_component = await app.update_component("as_llm", "default", model=old_model)
        await app.start()
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        replacement = await app.replace_component(
            "as_llm",
            "default",
            config={
                "backend": "dashscope",
                "model": "consumer-injected",
                "credential": {"api_key": ""},
            },
            runtime_updates={"model": verified_model},
        )

        assert isinstance(replacement, DashScopeAsLLM)
        assert replacement.model is verified_model
        assert replacement.is_started is True
        assert old_component.is_started is False
        assert wrapper.as_llm is replacement
        assert app.context.components[ComponentEnum.AS_LLM]["default"] is replacement
        assert app.config.components[ComponentEnum.AS_LLM]["default"].backend == "dashscope"
        assert old_component not in app._started_components
        assert replacement in app._started_components
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_before_start_preserves_dependency_order(tmp_path):
    """Unresolved bind placeholders continue to resolve during normal startup."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    verified_model = object()

    async def exercise_api() -> None:
        replacement = await app.replace_component(
            "as_llm",
            "default",
            config={
                "backend": "dashscope",
                "model": "consumer-injected",
                "credential": {"api_key": ""},
            },
            runtime_updates={"model": verified_model},
        )
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        assert wrapper.dependencies[0].name == "default"
        await app.start()
        assert wrapper.as_llm is replacement
        assert replacement.is_started is True
        await app.close()

    asyncio.run(exercise_api())


def test_replace_component_start_failure_keeps_old_generation(tmp_path):
    """Construction/start failures do not expose a partially replaced graph."""

    class BrokenAsLLM(BaseAsLLM):
        """Backend whose startup deterministically fails for rollback tests."""

        component_type = ComponentEnum.AS_LLM

        async def _start(self) -> None:
            raise RuntimeError("replacement failed")

    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    app.context.registry.add("broken", BrokenAsLLM, owner="test")

    async def exercise_api() -> None:
        old_component = await app.update_component("as_llm", "default", model=object())
        await app.start()
        wrapper = app.context.components[ComponentEnum.AGENT_WRAPPER]["default"]

        with pytest.raises(RuntimeError, match="replacement failed"):
            await app.replace_component(
                "as_llm",
                "default",
                config={
                    "backend": "broken",
                    "model": "unused",
                    "credential": {},
                },
            )

        assert app.context.components[ComponentEnum.AS_LLM]["default"] is old_component
        assert app.config.components[ComponentEnum.AS_LLM]["default"].backend == "openai"
        assert wrapper.as_llm is old_component
        assert old_component.is_started is True
        assert old_component in app._started_components
        await app.close()

    asyncio.run(exercise_api())
