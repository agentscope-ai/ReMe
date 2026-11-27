"""Russian tokenizer with optional lemmatization."""

import re
from typing import Callable

from .regex_tokenizer import RegexTokenizer
from ...core.registry import R


@R.register("russian")
class RussianTokenizer(RegexTokenizer):
    """Regex tokenizer that folds Cyrillic words to a shared form.

    Russian keyword search misses related word forms whenever a note and a query
    use different cases or numbers: a note containing ``инструментом`` does not
    match a query for ``инструмента``. Folding Cyrillic tokens to a common form
    lets both sides land on the same index term.

    Two names are involved, and they mean different things:

    - ``backend`` is the registry key used by ``Application`` to pick this class,
      so it stays ``russian``. It is not a choice the user makes about tokens.
    - ``engine`` selects the folding implementation:

      - ``snowball`` (default): the Snowball stemmer. A pure algorithm with no
        dictionaries, adding roughly 0.2 MiB. It returns a *stem*, not a
        dictionary form -- ``система`` becomes ``систем``. That is sufficient
        for retrieval, but the stem is not a real word and must not be shown to
        users.
      - ``pymorphy3``: a morphological analyzer returning the dictionary form --
        ``система`` stays ``система`` -- at the cost of a 16 MiB Russian
        dictionary.

    Non-Cyrillic tokens, CJK included, behave exactly like ``RegexTokenizer``.
    """

    SUPPORTED_ENGINES = ("snowball", "pymorphy3")

    CYRILLIC_PATTERN = re.compile(r"[а-яё]", re.IGNORECASE)

    def __init__(self, backend: str = "russian", engine: str = "snowball", **kwargs):
        super().__init__(backend=backend, **kwargs)
        engine = (engine or "snowball").strip().lower()
        if engine not in self.SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported Russian tokenizer engine: {engine!r}. Expected one of {self.SUPPORTED_ENGINES}.",
            )
        self.engine = engine
        self._normalize: Callable[[str], str] | None = None

    @property
    def token_output_engine(self) -> str | None:
        """Snowball and pymorphy3 emit different forms, so the index must know which one ran."""
        return self.engine

    async def _start(self) -> None:
        await super()._start()
        if self.engine == "snowball":
            try:
                import snowballstemmer
            except ImportError as exc:
                raise ImportError(
                    "RussianTokenizer with engine='snowball' requires snowballstemmer. "
                    "Install it with: pip install 'reme-ai[russian]'",
                ) from exc
            self._normalize = snowballstemmer.stemmer("russian").stemWord
        else:
            try:
                import pymorphy3
            except ImportError as exc:
                raise ImportError(
                    "RussianTokenizer with engine='pymorphy3' requires pymorphy3. "
                    "Install it with: pip install 'reme-ai[russian-full]'",
                ) from exc
            analyzer = pymorphy3.MorphAnalyzer()
            self._normalize = lambda word: analyzer.parse(word)[0].normal_form  # noqa: E731

        # Stopwords are compared against the tokens this tokenizer emits, and
        # those are already folded by the time `_postprocess` filters them. The
        # set has to be folded with the same engine, or an exact word stops
        # matching: `машины` sits in the file, `машин` is what the comparison
        # actually sees, and the stopword survives.
        if self._stopwords:
            self._stopwords = {self._fold(word) for word in self._stopwords}
        self.logger.info(f"RussianTokenizer started with engine={self.engine}")

    async def _close(self) -> None:
        self._normalize = None
        await super()._close()

    def _fold(self, token: str) -> str:
        """Fold a Cyrillic token; every other token is returned unchanged.

        Lowercasing belongs to ``BaseTokenizer._postprocess()``, which honours
        the caller's ``lower`` argument. Lowering here as well dropped the case
        of Latin tokens even when the caller asked to keep it.
        """
        if self._normalize is None:
            return token
        lowered = token.lower()
        if not self.CYRILLIC_PATTERN.search(lowered):
            return token
        return self._normalize(lowered)

    def _tokenize_one(self, text: str, **kwargs) -> list[str]:
        tokens = super()._tokenize_one(text, **kwargs)
        if self._normalize is None:
            # Not started yet: behave like the plain regex tokenizer.
            return tokens
        return [self._fold(token) for token in tokens]
