"""``digest_edit_step`` — constrained EditStep that targets ``digest/<bucket>/<slug>.md``.

Subclasses :class:`EditStep`. On top of the generic find-and-replace,
this step adds:

* **Path-shape validation** — the target must be ``digest/<bucket>/<slug>.md``
  with ``<bucket>`` in the configured set.
* **Existence check** — the file must already exist (use
  :class:`DigestWriteStep` for new nodes).
* **E-1 strong edge conservation** — the wikilink set in the file BEFORE
  the replacement must be a subset of the wikilink set AFTER. The check
  runs as a preflight: the replacement is simulated, links are compared,
  and the actual write is delegated to ``super().execute()`` only if
  conservation holds. On violation the step returns
  ``REJECT_CONSERVATION`` (with the missing edges) and the file on disk
  is left untouched.
"""

from pathlib import Path

import frontmatter

from .digest_write import _validate_digest_path, bucket_names, normalize_buckets
from ...file_io._file_io import read_file_safe
from ...file_io.edit import EditStep
from ....components import R
from ....utils.wikilink_handler import WikilinkHandler


@R.register("digest_edit_step")
class DigestEditStep(EditStep):
    """EditStep variant that enforces digest path shape + E-1 edge conservation.

    ``digest_dir`` is sourced from ``app_context.app_config.digest_dir`` at
    execute time — same convention as the ``daily_*`` steps.
    """

    def __init__(self, buckets=None, **kwargs):
        super().__init__(**kwargs)
        self.buckets = normalize_buckets(buckets)

    def _reject(self, message: str, **meta) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"REJECT: {message}"
        if meta:
            self.context.response.metadata.update(meta)

    async def execute(self):
        assert self.context is not None
        raw = str(self.context.get("path") or "")
        old = self.context.get("old")
        new = self.context.get("new")
        digest_dir = getattr(self.app_context.app_config, "digest_dir", "")

        err = _validate_digest_path(raw, bucket_names(self.buckets), digest_dir)
        if err:
            self._reject(err)
            return None

        abs_path = (Path(self.vault_path) / raw).resolve()
        if not abs_path.exists():
            self._reject(f"{raw} does not exist; use digest_write instead")
            return None

        # Preflight conservation check: simulate the find-and-replace, compare
        # wikilink sets before vs after, refuse the write if any edge is dropped.
        # We let super().execute() re-validate `old in body` and surface its
        # own error if old is empty or missing.
        if old is not None and new is not None and str(old) != "":
            preview_text = await _preview_replacement(abs_path, str(old), str(new))
            if preview_text is not None:
                missing = _missing_edges(
                    await _read_text(abs_path),
                    preview_text,
                    raw,
                )
                if missing:
                    missing_repr = sorted(f"[[{t}]]" + (f" (predicate={p})" if p else "") for t, p in missing)
                    self._reject(
                        (
                            f"REJECT_CONSERVATION: replacement drops {len(missing)} edge(s) "
                            "the old body had. E-1 strong-conservation: every outbound wikilink "
                            "in the old body MUST appear in the new body. "
                            f"Missing: {', '.join(missing_repr)}. "
                            "Adjust `new` to keep the missing links and retry."
                        ),
                        conservation_violation={
                            "target_path": raw,
                            "missing": sorted(list(missing)),
                        },
                    )
                    return None

        return await super().execute()


async def _read_text(abs_path: Path) -> str:
    text, _ = await read_file_safe(abs_path)
    return text


async def _preview_replacement(abs_path: Path, old: str, new: str) -> str | None:
    """Return the full file text as it WOULD look after EditStep's replace.

    Mirrors EditStep's body-only replacement (frontmatter is left untouched).
    Returns ``None`` if the replacement isn't possible (``old`` not in body) so
    the caller can defer the error to super().execute()."""
    raw_text, _ = await read_file_safe(abs_path)
    post = frontmatter.loads(raw_text)
    body = post.content
    if old not in body:
        return None
    post.content = body.replace(old, new)
    new_text = frontmatter.dumps(post) if post.metadata else post.content
    if not new_text.endswith("\n"):
        new_text += "\n"
    return new_text


def _missing_edges(old_text: str, new_text: str, path: str) -> set[tuple[str, str | None]]:
    """Edges present in ``old_text`` but absent from ``new_text`` (E-1 deltas)."""
    old_set = {(link.target_path, link.predicate) for link in WikilinkHandler.extract_links(old_text, path)}
    new_set = {(link.target_path, link.predicate) for link in WikilinkHandler.extract_links(new_text, path)}
    return old_set - new_set
