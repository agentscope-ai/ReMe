"""Module for registering AgentScope token counters."""

from .reme_token_counter import ReMeTokenCounter
from ..registry_factory import R

R.as_token_counters.register("hf")(ReMeTokenCounter)
