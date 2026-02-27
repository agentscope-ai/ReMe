"""token counter"""

from .base_token_counter import BaseTokenCounter
from .hf_token_counter import HFTokenCounter
from .openai_token_counter import OpenAITokenCounter
from ..registry_factory import R

__all__ = [
    "BaseTokenCounter",
    "HFTokenCounter",
    "OpenAITokenCounter",
]

R.token_counters.register("base")(BaseTokenCounter)
R.token_counters.register("hf")(HFTokenCounter)
R.token_counters.register("openai")(OpenAITokenCounter)
