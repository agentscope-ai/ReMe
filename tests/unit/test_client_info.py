"""The desktop discovers public capabilities without depending on backend wiring."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace

import pytest

from reme.components import ApplicationContext
from reme.schema.agent_view import ClientInfo
from reme.steps.common.app_config import ClientInfoStep


@pytest.mark.asyncio
async def test_client_info_respects_service_jobs_and_exposes_no_component_configuration(tmp_path):
    context = ApplicationContext(
        workspace_dir=str(tmp_path),
        jobs={
            "ask_agent": {"backend": "stream"},
            "asr": {"backend": "base"},
            "tts": {"backend": "base"},
            "disabled": {"enabled": False},
            "private": {"enable_serve": False},
            "worker": {"backend": "background"},
        },
        components={
            "file_store": {"default": {"tag_index": "custom"}},
            "tag_index": {"custom": {"tag_key": "topics"}},
            "tts": {"default": {"model": "voice-model", "voice": "voice-1", "api_key": "secret"}},
        },
    )
    step = ClientInfoStep(app_context=context)
    value = ClientInfo.model_validate((await step()).answer)
    assert value.jobs == ["ask_agent", "asr", "tts"]
    assert value.asr and value.tts and value.tag_key == "topics"
    assert value.workspace_dir == str(tmp_path) and value.daily_dir == "daily"
    assert "secret" not in value.model_dump_json() and "components" not in value.model_dump_json()
    context.service = SimpleNamespace(jobs={"ask_agent"})
    limited = ClientInfo.model_validate((await step()).answer)
    assert limited.jobs == ["ask_agent"] and not limited.asr and not limited.tts
    assert limited.speech_revision == value.speech_revision
    context.app_config.components["tts"]["default"].voice = "voice-2"
    assert (await step()).answer["speech_revision"] != value.speech_revision
