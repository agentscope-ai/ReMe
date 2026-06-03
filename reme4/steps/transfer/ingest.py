"""``ingest`` — capture an externally-received asset into ``resource/<date>/``.

This step is the dedicated information-capture interface: when an
external channel (wechat group, email, browser save, API push, ...)
hands the agent a file, ``ingest`` lands it in ``resource/<YYYY-MM-DD>/``
keyed by the day it was received (always today, local time),
alongside a ``meta.json`` row recording provenance.

This is the only ingest path through ``resource/``. Materials the
main agent *actively* fetches or generates during a daily task
belong inside the daily folder as sibling materials, not here.

For generic file copy (local fs → arbitrary vault path) use
``upload`` instead.

Bucket layout::

    resource/<YYYY-MM-DD>/
      meta.json                                 # JSON array of FileNode rows
      <date>.md                                 # derived markdown view
      <channel>__<HHMMSS>__<source-basename>    # the asset itself
      ...

Naming convention — the file name is **derived**, never caller-supplied::

    <channel>__<HHMMSS>__<source-basename>

* ``<channel>`` — top-level identity (anchors provenance), lowercase
  letters/digits/dashes only.
* ``<HHMMSS>`` — receive time within the bucket day (the date is
  already implicit in the bucket folder).
* ``<source-basename>`` — the basename of the ``path`` argument,
  after rejecting path separators, dot segments, and leading dots.

This format is self-describing in directory listings + wikilinks
(``[[resource/<date>/wechat__153022__report.pdf]]``) and makes
cross-channel basename collisions structurally impossible. Two
genuine duplicates (same channel + same second + same basename)
are reported as an error — the step never silently dedupes, so
callers see the conflict and can decide whether to retry, rename
upstream, or skip.

Each ``ingest`` call:

1. Resolves the bucket date as today (local time).
2. Validates the inputs: ``path`` exists, ``channel`` matches the
   allowed character class, ``description`` is non-empty, the
   source basename has no path separators / dot segments / leading
   dot.
3. Builds the final name from the format above.
4. Under a per-day file lock, checks the final name against
   ``meta.json`` ∪ the on-disk listing. Any collision → error
   (no silent suffixing).
5. Copies the asset into the bucket.
6. Appends a :class:`FileNode` row to ``meta.json`` (with provenance
   on ``front_matter``) and re-renders ``<date>.md``.

Parameters:

* ``path`` (required) — local filesystem path to the asset to ingest.
* ``channel`` (required) — inbound channel identifier (wechat /
  email / browser / api / ...). Lowercase letters / digits / dashes
  only.
* ``description`` (required) — analysis hint for downstream agents:
  where the asset came from, what kind of content it carries, and how
  it should be interpreted. The dreamer / auto_memory reads this
  verbatim from ``meta.json`` to decide how to read the asset (skim
  vs. deep parse, structured extraction vs. summarization, etc.), so
  callers should write enough detail to drive that decision — not
  just a title. Multi-line is fine; the ``<date>.md`` bullet view
  flattens for display while ``meta.json`` preserves the original.
* ``metadata`` (optional dict) — extras persisted on the meta row.
  ``source`` (free-form origin within the channel) is conventional;
  any other keys pass through verbatim. Keys ``name``, ``channel``,
  ``received_at``, ``description`` are reserved.

Returns ``{date, name, path}`` on success or ``{error}`` on failure.
"""

import datetime
import fcntl
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from ..base_step import BaseStep

from ...components import R

from ...schema import FileFrontMatter, FileNode


# Channel identifier character class — keeps the derived filename predictable
# and parseable (the `__` separator is also disjoint from this set).
_CHANNEL_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# Keys the step manages itself — callers cannot smuggle them in via metadata.
_RESERVED_METADATA_KEYS = frozenset({"name", "channel", "received_at", "description"})


@R.register("ingest_step")
class IngestStep(BaseStep):
    """Capture an external asset into ``resource/<date>/`` and update the day's meta + index."""

    async def execute(self):
        assert self.context is not None
        path: str = (self.context.get("path", "") or "").strip()
        channel: str = (self.context.get("channel", "") or "").strip()
        description: str = (self.context.get("description", "") or "").strip()
        metadata_raw = self.context.get("metadata") or {}

        prepared, prep_error = _prepare_inputs(path, channel, description, metadata_raw)
        if prep_error:
            self._fail({"error": prep_error})
            self.logger.info(f"[{self.name}] ingest failed channel={channel!r} error={prep_error!r}")
            return

        try:
            outcome = self._land(
                src=prepared["src"],
                date=prepared["date"],
                final_name=prepared["final_name"],
                entry_fields={
                    **prepared["metadata"],
                    "channel": channel,
                    "received_at": prepared["received_at"],
                    "description": description,
                },
            )
        except _DuplicateIngest as e:
            self._fail({"error": str(e)})
            self.logger.info(f"[{self.name}] ingest duplicate channel={channel!r} error={str(e)!r}")
            return
        except Exception as e:
            self._fail({"error": f"{type(e).__name__}: {e}"})
            self.logger.info(f"[{self.name}] ingest crashed channel={channel!r} error={type(e).__name__}: {e}")
            return

        self.context.response.success = True
        self.context.response.answer = f"Ingested {outcome['name']} to {outcome['path']}"
        self.context.response.metadata.update(outcome)
        self.logger.info(
            f"[{self.name}] channel={channel} date={outcome['date']} name={outcome['name']} path={outcome['path']}",
        )

    # ------------------------------------------------------------------

    def _fail(self, payload: dict) -> None:
        assert self.context is not None
        self.context.response.success = False
        self.context.response.answer = f"Error: {payload.get('error', 'ingest failed')}"
        self.context.response.metadata.update(payload)

    def _resource_dir_name(self) -> str:
        """Configured ``resource_dir`` subdir name; defaults to ``"resource"`` outside an app context."""
        if self.app_context is None:
            return "resource"
        return self.app_context.app_config.resource_dir

    def _vault_dir(self) -> Path:
        vr = getattr(self.file_store, "vault_path", None)
        return Path(vr).resolve() if vr else Path.cwd().resolve()

    def _land(self, src: Path, date: str, final_name: str, entry_fields: dict) -> dict:
        resource_dir = self._resource_dir_name()
        bucket = self._vault_dir() / resource_dir / date
        bucket.mkdir(parents=True, exist_ok=True)
        meta_path = bucket / "meta.json"
        day_md = bucket / f"{date}.md"
        lock_path = bucket / ".lock"

        rel_path = f"{resource_dir}/{date}/{final_name}"

        with _bucket_lock(lock_path):
            existing_entries = _read_meta(meta_path)
            # Collision check spans meta ∪ on-disk listing so a stray file
            # (from a crashed earlier run) and case-insensitive filesystems
            # both surface the conflict rather than getting silently clobbered.
            on_disk = {p.name for p in bucket.iterdir() if p.is_file()}
            existing_names = {Path(e.path).name for e in existing_entries} | on_disk
            if final_name in existing_names:
                raise _DuplicateIngest(
                    f"duplicate: {final_name!r} already exists in {resource_dir}/{date}/",
                )

            dst_path = bucket / final_name
            shutil.copyfile(src, dst_path)

            # FileFrontMatter has first-class `name` / `description`; everything
            # else (channel / source / received_at / passthrough metadata) rides
            # the extras bag (model_config extra="allow").
            description = entry_fields.pop("description", "")
            entry = FileNode(
                path=rel_path,
                st_mtime=dst_path.stat().st_mtime,
                front_matter=FileFrontMatter(description=description, **entry_fields),
            )
            updated = existing_entries + [entry]
            _atomic_write_text(
                meta_path,
                json.dumps([e.model_dump() for e in updated], ensure_ascii=False, indent=2) + "\n",
            )
            _atomic_write_text(day_md, _assemble_day_md(updated, date))

        return {
            "date": date,
            "name": final_name,
            "path": rel_path,
        }


class _DuplicateIngest(Exception):
    """Raised when the derived name already exists in the bucket."""


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------


def _prepare_inputs(
    path: str,
    channel: str,
    description: str,
    metadata_raw,
) -> tuple[dict, str]:
    """Validate caller args and derive the bucket date / final name.

    Returns ``(prepared, error)``: on success ``prepared`` has
    ``{src, date, received_at, final_name, metadata}`` and ``error`` is
    empty; on failure ``prepared`` is ``{}`` and ``error`` carries the
    first violation in user-facing order (path → channel → description
    → metadata-shape → file existence → basename → metadata-keys).
    """
    src = Path(path) if path else None
    metadata, meta_error = _sanitize_metadata(metadata_raw) if isinstance(metadata_raw, dict) else ({}, "")
    error = next(
        (
            msg
            for msg in (
                "path is required" if not path else "",
                _validate_channel(channel),
                "description is required" if not description else "",
                "metadata must be a dict" if not isinstance(metadata_raw, dict) else "",
                f"path not found: {path}" if src is not None and not src.is_file() else "",
                _validate_basename(src.name) if src is not None else "",
                meta_error,
            )
            if msg
        ),
        "",
    )
    if error:
        return {}, error

    assert src is not None  # narrowed by the "path is required" check
    now = datetime.datetime.now()
    return (
        {
            "src": src,
            "date": now.strftime("%Y-%m-%d"),
            "received_at": now.isoformat(timespec="seconds"),
            "final_name": f"{channel}__{now.strftime('%H%M%S')}__{src.name}",
            "metadata": metadata,
        },
        "",
    )


def _validate_channel(channel: str) -> str:
    """Return an error string when ``channel`` is unsafe; empty when OK."""
    if not channel:
        return "channel is required"
    if not _CHANNEL_RE.match(channel):
        return f"channel {channel!r} must be lowercase letters / digits / dashes " f"and start with a letter or digit"
    return ""


def _validate_basename(name: str) -> str:
    """Return an error string when the source basename is unsafe; empty when OK.

    The derived filename embeds this string after a ``__`` separator, so we
    only need to block characters that would mangle the filesystem path —
    path separators, dot segments, leading dots. Bookkeeping-name collisions
    (``meta.json`` / ``<date>.md``) are impossible by construction once the
    channel + time prefix is prepended.
    """
    if not name:
        return "source basename is empty"
    if "/" in name or "\\" in name:
        return f"source basename must not contain path separators: {name!r}"
    if name in {".", ".."}:
        return f"source basename {name!r} is reserved"
    if name.startswith("."):
        return f"source basename may not start with '.': {name!r}"
    return ""


def _sanitize_metadata(raw: dict) -> tuple[dict, str]:
    """Return ``(cleaned_metadata, error)``.

    Rejects reserved keys (those the step manages itself) and coerces
    ``source`` to a string. All other keys pass through verbatim so
    callers can attach arbitrary tags that land on the entry's
    ``front_matter`` extras.
    """
    for key in _RESERVED_METADATA_KEYS:
        if key in raw:
            return {}, f"metadata key {key!r} is reserved"

    cleaned = dict(raw)
    if "source" in cleaned:
        src = cleaned["source"]
        cleaned["source"] = src.strip() if isinstance(src, str) else ""
    return cleaned, ""


# ----------------------------------------------------------------------
# Per-day exclusive lock + atomic write helpers
# ----------------------------------------------------------------------


class _bucket_lock:
    """Exclusive ``flock`` on a per-day lock file; serializes meta+md writes."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fp = None

    def __enter__(self):
        self._fp = open(self.lock_path, "w", encoding="utf-8")
        fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fp is not None:
            try:
                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            finally:
                self._fp.close()
                self._fp = None


def _read_meta(meta_path: Path) -> list[FileNode]:
    """Read ``meta.json`` as a list of :class:`FileNode` rows; missing or malformed → []."""
    if not meta_path.is_file():
        return []
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[FileNode] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        try:
            out.append(FileNode(**row))
        except Exception:
            continue
    return out


def _atomic_write_text(target: Path, text: str) -> None:
    """Atomic text write via tempfile + os.replace in the same directory."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".ingest-", dir=target.parent)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ----------------------------------------------------------------------
# Day-view markdown rendering (pure)
# ----------------------------------------------------------------------


def _assemble_day_md(entries: list[FileNode], date: str) -> str:
    """Render the day's bucket as a markdown view.

    Layout::

        ---
        name: <date>
        assets: [<name1>, <name2>, ...]
        ---

        # <date> resources

        - [[<path>]] — <channel> from `<source>` at <hh:mm> — <description>

    Provenance lives on each entry's ``front_matter`` (``channel`` /
    ``source`` / ``received_at`` as extras, ``description`` as the
    first-class field). ``received_at`` is rendered as ``HH:MM`` when it
    parses as ISO 8601, else dropped silently — the asset list stays
    readable even when upstream channels emit malformed timestamps.
    ``source`` is dropped when empty. ``description`` is flattened
    (newlines collapsed to spaces) so the bullet stays one line per
    asset; ``meta.json`` preserves the verbatim multi-line text.
    """
    names = [Path(e.path).name for e in entries]
    lines: list[str] = [
        "---",
        f"name: {date}",
        f"assets: [{', '.join(names)}]",
        "---",
        "",
        f"# {date} resources",
        "",
    ]
    for entry in entries:
        fm = entry.front_matter
        channel = getattr(fm, "channel", "") or ""
        source = getattr(fm, "source", "") or ""
        received_at = getattr(fm, "received_at", "") or ""
        description = fm.description or ""

        bits: list[str] = [f"- [[{entry.path}]]"]
        provenance = channel
        if source:
            provenance += f" from `{source}`"
        time_part = _hhmm(received_at)
        if time_part:
            provenance += f" at {time_part}"
        bits.append(provenance)
        if description:
            # Flatten so the bullet stays one line per asset; meta.json keeps
            # the verbatim multi-line description for downstream consumers.
            bits.append(" ".join(description.split()))
        lines.append(" — ".join(bits))
    return "\n".join(lines) + "\n"


def _hhmm(received_at: str) -> str:
    """Best-effort HH:MM extraction from an ISO 8601 timestamp."""
    if not received_at:
        return ""
    raw = received_at.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(raw).strftime("%H:%M")
    except ValueError:
        return ""
