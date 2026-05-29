"""``digest_write_step`` — constrained WriteStep that targets ``digest/<bucket>/<slug>.md``.

Subclasses :class:`WriteStep`. The only thing this step adds on top of the
generic file write is path-shape validation:

* ``path`` must look like ``digest/<bucket>/<slug>.md`` (depth 1, ``.md`` suffix).
* ``<bucket>`` must be one of the configured ``buckets`` (defaults to the
  dreamer's :data:`DEFAULT_BUCKETS`).
* The target must NOT already exist — for in-place updates the agent uses
  :class:`DigestEditStep` instead.

Everything else (atomic write semantics, encoding, parent-dir creation,
frontmatter handling) is inherited from :class:`WriteStep`.

Bucket vocabulary lives in this module (NOT in the dreamer prompt template)
so it can be swapped / extended without touching the prompt. Future direction:
externalize via app-config injection; the ``Dreamer`` constructor already
accepts an override.
"""

from pathlib import Path

from ..file_io.write import WriteStep
from ...components import R


# Each bucket carries a name (the filesystem folder under ``digest/``) and a
# one-line description that the Phase 2 prompt renders into the bucket-picking
# heuristic block at runtime.
DEFAULT_BUCKETS: tuple[dict[str, str], ...] = (
    {
        "name": "concept",
        "description": 'definitions, principles, mental models ("what IS X?")',
    },
    {
        "name": "procedure",
        "description": 'steps, methods, recipes ("how do I do X?")',
    },
    {
        "name": "entity",
        "description": "specific named things (person, system, tool, project)",
    },
    {
        "name": "observation",
        "description": 'findings, results, decisions with rationale ("what happened / was decided?")',
    },
    {
        "name": "preference",
        "description": 'user / team / agent collaboration rules ("how does X like to work / what to avoid?")',
    },
    {
        "name": "unknown",
        "description": "fallback when no specialized bucket fits (first-class, not a failure state)",
    },
)


def normalize_buckets(buckets) -> tuple[dict[str, str], ...]:
    """Normalize a caller-supplied bucket spec into ``tuple[{name, description}, ...]``.

    Accepts ``None`` (→ :data:`DEFAULT_BUCKETS`), tuple/list of dicts (returned
    as-is), or tuple/list of strings (legacy — each wrapped with an empty
    description).
    """
    if not buckets:
        return DEFAULT_BUCKETS
    first = next(iter(buckets))
    if isinstance(first, dict):
        return tuple(buckets)
    return tuple({"name": str(b), "description": ""} for b in buckets)


def bucket_names(buckets) -> tuple[str, ...]:
    """Extract just the name field of each bucket — used for path-membership checks."""
    if not buckets:
        return ()
    first = next(iter(buckets))
    if isinstance(first, dict):
        return tuple(b["name"] for b in buckets)
    return tuple(buckets)


@R.register("digest_write_step")
class DigestWriteStep(WriteStep):
    """WriteStep variant that enforces the ``<digest_dir>/<bucket>/<slug>.md`` layout.

    ``digest_dir`` is sourced from ``app_context.app_config.digest_dir`` at
    execute time — same convention as ``daily_create`` / ``daily_list`` /
    ``daily_reindex`` for their ``daily_dir``.
    """

    def __init__(self, buckets=None, **kwargs):
        super().__init__(**kwargs)
        self.buckets = normalize_buckets(buckets)

    def _reject(self, message: str) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"REJECT: {message}"

    async def execute(self):
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")

        err = _validate_digest_path(raw, bucket_names(self.buckets), digest_dir)
        if err:
            self._reject(err)
            return None

        abs_path = (Path(self.vault_path) / raw).resolve()
        if abs_path.exists():
            self._reject(f"{raw} already exists; use digest_edit instead")
            return None

        return await super().execute()


def _validate_digest_path(
    raw: str,
    allowed_names: tuple[str, ...],
    digest_dir: str,
) -> str | None:
    """Return an error message, or ``None`` if ``raw`` matches the digest shape
    and uses one of the allowed bucket names. ``digest_dir`` is the configured
    digest root (e.g. ``"digest"``)."""
    prefix = f"{digest_dir}/"
    expected = f"{digest_dir}/<bucket>/<slug>.md"
    if not raw.startswith(prefix) or not raw.endswith(".md"):
        return f"path must be {expected!r}, got {raw!r}"
    parts = raw.split("/")
    if len(parts) != 3:
        return f"digest is a shallow bucket layout ({expected}, depth 1); got {raw!r}"
    bucket = parts[1]
    if bucket not in allowed_names:
        return (
            f"bucket {bucket!r} not in allowed set {list(allowed_names)}. "
            "Use 'unknown' when no specialized bucket fits."
        )
    return None
