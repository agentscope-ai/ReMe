"""Tests for the application config step."""

import asyncio

from reme.components.application_context import ApplicationContext
from reme.steps.common.app_config import AppConfigStep


def test_app_config_step_returns_effective_config_without_secrets(tmp_path):
    """Effective values are returned while startup secrets stay private."""
    context = ApplicationContext(
        app_name="Test ReMe",
        workspace_dir=str(tmp_path),
        environment={"PRIVATE_VALUE": "hidden"},
        components={
            "as_llm": {
                "default": {
                    "backend": "openai",
                    "credential": {"api_key": "key", "base_url": "https://example.test"},
                },
            },
        },
        mcp_servers={
            "private": {
                "url": "https://user:password@example.test/mcp?access_token=url-token&mode=read",
                "headers": {
                    "Authorization": "Bearer header-token",
                    "X-API-Key": "header-key",
                },
                "client_secret": "client-secret",
            },
        },
    )

    response = asyncio.run(AppConfigStep(app_context=context)())

    assert response.answer["app_name"] == "Test ReMe"
    assert "environment" not in response.answer
    credential = response.answer["components"]["as_llm"]["default"]["credential"]
    assert credential == {"api_key": "***", "base_url": "https://example.test"}
    mcp_server = response.answer["mcp_servers"]["private"]
    assert mcp_server["url"] == "https://user:***@example.test/mcp?access_token=%2A%2A%2A&mode=read"
    assert mcp_server["headers"] == {"Authorization": "***", "X-API-Key": "***"}
    assert mcp_server["client_secret"] == "***"
