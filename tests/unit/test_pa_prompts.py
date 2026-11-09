"""Localized proactive queries use the current input and delivery tools."""

# pylint: disable=missing-function-docstring,protected-access

import pytest

from reme.steps.proactive import ProactiveAgentStep


@pytest.mark.parametrize("language", ["en", "zh"])
def test_proactive_query_reads_tool_text_directly(pa_workspace, language):
    step = ProactiveAgentStep(app_context=pa_workspace.context, language=language)
    query = step.prompt_format(
        "query",
        memory_path=step._memory_path(),
        proactive_config_path=pa_workspace.runtime.proactive_config_path,
        max_turns=7,
        optional_hint="",
        hint_period="normal",
    )
    assert ("返回的正文" if language == "zh" else "returned text") in query
    assert "input_path" not in query
