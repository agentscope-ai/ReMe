"""llm"""

from .base_llm import BaseLLM
from .lite_llm import LiteLLM
from .lite_llm_sync import LiteLLMSync
from .openai_llm import OpenAILLM
from .openai_llm_sync import OpenAILLMSync
from .novita_llm import NovitaLLM, NovitaLLMSync
from ..registry_factory import R

__all__ = [
    "BaseLLM",
    "LiteLLM",
    "LiteLLMSync",
    "OpenAILLM",
    "OpenAILLMSync",
    "NovitaLLM",
    "NovitaLLMSync",
]

R.llms.register("litellm")(LiteLLM)
R.llms.register("litellm_sync")(LiteLLMSync)
R.llms.register("openai")(OpenAILLM)
R.llms.register("openai_sync")(OpenAILLMSync)
R.llms.register("novita")(NovitaLLM)
R.llms.register("novita_sync")(NovitaLLMSync)
