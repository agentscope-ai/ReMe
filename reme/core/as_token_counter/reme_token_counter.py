"""Token counter for ReMe."""

import os
from typing import Any

from agentscope.token import HuggingFaceTokenCounter

from ..utils import get_logger

logger = get_logger()


class ReMeTokenCounter(HuggingFaceTokenCounter):
    """Token counter for CoPaw with configurable tokenizer support.

    This class extends HuggingFaceTokenCounter to provide token counting
    functionality with support for both local and remote tokenizers,
    as well as HuggingFace mirror for users in China.

    Attributes:
        pretrained_model_name_or_path: The tokenizer model path or "default" for local tokenizer.
        use_mirror: Whether to use HuggingFace mirror.
        token_count_estimate_divisor: Divisor for token estimation.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        use_mirror: bool = True,
        token_count_estimate_divisor: float = 3.75,
        **kwargs,
    ):
        """Initialize the token counter with the specified configuration.

        Args:
            pretrained_model_name_or_path: The tokenizer model path.
            use_mirror: Whether to use the HuggingFace mirror
                (https://hf-mirror.com) for downloading tokenizers. Useful for
                users in China.
            token_count_estimate_divisor: Divisor for estimating tokens when
                tokenizer is unavailable. Defaults to 3.75.
            **kwargs: Additional keyword arguments passed to HuggingFaceTokenCounter.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_mirror = use_mirror
        self.token_count_estimate_divisor = token_count_estimate_divisor

        # Set HuggingFace endpoint for mirror support
        if use_mirror:
            mirror = "https://hf-mirror.com"
        else:
            mirror = "https://huggingface.co"

        os.environ["HF_ENDPOINT"] = mirror

        # if the huggingface is already imported in other dependencies,
        # we need to set the endpoint manually
        import huggingface_hub.constants

        huggingface_hub.constants.ENDPOINT = mirror
        huggingface_hub.constants.HUGGINGFACE_CO_URL_TEMPLATE = mirror + "/{repo_id}/resolve/{revision}/{filename}"

        try:
            super().__init__(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                use_mirror=use_mirror,
                use_fast=True,
                trust_remote_code=True,
                **kwargs,
            )
            self._tokenizer_available = True

        except Exception as e:
            logger.error(f"Failed to initialize tokenizer {e}")
            self._tokenizer_available = False

    async def count(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        text: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Count tokens in messages or text.

        If text is provided, counts tokens directly in the text string.
        Otherwise, counts tokens in the messages using the parent class method.

        Args:
            messages: List of message dictionaries in chat format.
            tools: Optional list of tool definitions for token counting.
            text: Optional text string to count tokens directly.
            **kwargs: Additional keyword arguments passed to parent count method.

        Returns:
            The number of tokens, guaranteed to be at least the estimated minimum.
        """
        if text:
            if self._tokenizer_available:
                try:
                    token_ids = self.tokenizer.encode(text)
                    return max(len(token_ids), self.estimate_tokens(text))
                except Exception as e:
                    logger.exception("Failed to encode text with tokenizer: %s", e)
                    return self.estimate_tokens(text)
            else:
                return self.estimate_tokens(text)
        else:
            return await super().count(messages, tools, **kwargs)

    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.

        Provides a fast character-based estimation as a fallback or lower bound.
        Uses the configured divisor from instance settings.

        Args:
            text: The text string to estimate tokens for.

        Returns:
            The estimated number of tokens in the text string.
        """
        return int(len(text.encode("utf-8")) / self.token_count_estimate_divisor + 0.5)
