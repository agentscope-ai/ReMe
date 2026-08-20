"""Tests for the OrcaRouter LLM provider."""

# pylint: disable=missing-function-docstring,missing-class-docstring,unused-argument

from reme.components.as_llm import OrcaRouterAsLLM, OrcaRouterCredential
from reme.components.component_registry import R, create_application_registry
from reme.enumeration import ComponentEnum


def test_orcarouter_registered_in_builtin_registry():
    assert R.get(ComponentEnum.AS_LLM, "orcarouter") is OrcaRouterAsLLM


def test_application_registry_includes_orcarouter():
    reg = create_application_registry()
    assert reg.get(ComponentEnum.AS_LLM, "orcarouter") is OrcaRouterAsLLM


def test_orcarouter_credential_defaults_to_hosted_gateway():
    credential = OrcaRouterCredential(api_key="sk-orca-test")
    assert credential.base_url == "https://api.orcarouter.ai/v1"


def test_orcarouter_credential_empty_base_url_falls_back_to_gateway():
    # The shared config passes ``base_url: ${LLM_BASE_URL:-}``, which yields an
    # empty string when LLM_BASE_URL is unset.
    credential = OrcaRouterCredential(api_key="sk-orca-test", base_url="")
    assert credential.base_url == "https://api.orcarouter.ai/v1"


def test_orcarouter_credential_custom_base_url_kept():
    credential = OrcaRouterCredential(
        api_key="sk-orca-test",
        base_url="https://selfhosted.example.com/v1",
    )
    assert credential.base_url == "https://selfhosted.example.com/v1"


def test_orcarouter_credential_reuses_openai_chat_model():
    assert OrcaRouterCredential.get_chat_model_class().__name__ == "OpenAIChatModel"
