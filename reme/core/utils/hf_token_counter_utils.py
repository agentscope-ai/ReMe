"""Utility functions for working with text."""

from agentscope.token import HuggingFaceTokenCounter

_token_counter = None


def get_hf_token_counter(
    pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    use_mirror=True,
    use_fast=True,
    trust_remote_code=True,
):
    """Get or initialize the global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = HuggingFaceTokenCounter(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_mirror=use_mirror,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
        )
    return _token_counter
