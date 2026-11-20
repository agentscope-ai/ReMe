"""Local Qwen token counting and truncation utilities."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_TOKENIZER = "Qwen/Qwen3.8-27B"


def estimate_token_count(
    text: str,
    estimate_divisor: float = 4,
    encoding: str = "utf-8",
) -> int:
    """Estimate the number of tokens in *text* by byte length."""
    return int(len(text.encode(encoding)) / estimate_divisor + 0.5)


def _estimated_count(text: str) -> int:
    return max(1, estimate_token_count(text)) if text else 0


class QwenTokenTools:
    """Count and truncate text with a lazily loaded Hugging Face tokenizer.

    Install ``reme-ai[qwen-tokenizer]`` for exact counts. The default source
    downloads tokenizer data only; select the inference model's source when
    model-specific token counts are needed. Truncation preserves a prefix of
    the original string, including when the tokenizer normalizes its input.
    Injected Hugging Face tokenizers are copied with truncation and padding disabled.
    Local sources are read on first use for each instance, so new instances see file updates.
    """

    def __init__(self, tokenizer_source: str = DEFAULT_TOKENIZER, *, tokenizer: Any | None = None) -> None:
        if not isinstance(tokenizer_source, str) or not tokenizer_source.strip():
            raise ValueError("tokenizer_source must be a non-empty string")
        self.tokenizer_source = tokenizer_source.strip()
        if tokenizer is not None:
            try:
                from tokenizers import Tokenizer
                from tokenizers.implementations import BaseTokenizer
            except ImportError:
                pass  # Custom tokenizer implementations do not require this package.
            else:
                if isinstance(tokenizer, (Tokenizer, BaseTokenizer)):
                    tokenizer = Tokenizer.from_str(tokenizer.to_str())
                    tokenizer.no_truncation()
                    tokenizer.no_padding()
        self._tokenizer = tokenizer
        self._tokenizer_lock = Lock()
        self._tokenizer_unavailable = False

    @property
    def tokenizer(self) -> Any | None:
        """Load once under a lock; warn once and return None if unavailable."""
        with self._tokenizer_lock:
            if self._tokenizer_unavailable:
                return None
            try:
                if self._tokenizer is None:
                    self._tokenizer = _load_tokenizer(self.tokenizer_source)
                return self._tokenizer
            except (ImportError, RuntimeError) as exc:
                self._tokenizer_unavailable = True
                reason = "missing 'tokenizers' package" if isinstance(exc, ImportError) else "load failed"
        warnings.warn(
            f"Qwen tokenizer is unavailable ({reason}); token count is estimated from UTF-8 byte length. "
            "Install exact tokenization support with: pip install 'reme-ai[qwen-tokenizer]'.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    def count(self, text: str) -> int:
        """Count tokenizer IDs, or estimate from bytes when the tokenizer is unavailable."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return 0
        tokenizer = self.tokenizer
        if tokenizer is None:
            return _estimated_count(text)
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> tuple[str, int, bool]:
        """Return an original-text prefix, its token count, and whether text changed."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not text:
            return "", 0, False
        tokenizer = self.tokenizer
        if tokenizer is None:
            token_count = _estimated_count(text)
            if token_count <= max_tokens:
                return text, token_count, False
            low, high = 0, len(text)
            while low < high:
                middle = (low + high + 1) // 2
                if _estimated_count(text[:middle]) <= max_tokens:
                    low = middle
                else:
                    high = middle - 1
            truncated = text[:low]
            return truncated, _estimated_count(truncated), True

        encoding = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(encoding.ids)
        if token_count <= max_tokens:
            return text, token_count, False

        # Byte-level tokens can share character offsets. Re-encode the original
        # prefix to check the budget instead of decoding a partial character.
        end = encoding.offsets[max_tokens - 1][1]
        truncated = text[:end]
        token_count = len(tokenizer.encode(truncated, add_special_tokens=False).ids)
        while truncated and token_count > max_tokens:
            truncated = truncated[:-1]
            token_count = len(tokenizer.encode(truncated, add_special_tokens=False).ids)
        return truncated, token_count, True


def _load_tokenizer(source: str) -> Any:
    """Read local files afresh; build independent instances from cached remote data."""
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError("Install 'reme-ai[qwen-tokenizer]' for exact tokenization") from exc
    try:
        local_path = Path(source)
        if local_path.is_dir():
            tokenizer = Tokenizer.from_file(str(local_path / "tokenizer.json"))
        elif local_path.is_file():
            tokenizer = Tokenizer.from_file(str(local_path))
        else:
            tokenizer = Tokenizer.from_str(_load_remote_tokenizer_json(source))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer
    except Exception as exc:
        raise RuntimeError(f"Unable to load Qwen tokenizer {source!r}: {exc}") from exc


@lru_cache(maxsize=4)
def _load_remote_tokenizer_json(source: str) -> str:
    """Cache remote serialized data, never local files or mutable tokenizer objects."""
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(source).to_str()
