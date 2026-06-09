"""AutoDreamStep — daily-tick wrapper that dispatches per-file to the ``dream`` job.

Each tick scans today's two surfaces under ``<daily_dir>/``:

* ``<daily_dir>/<today>.md`` — the day-index file (auto-rebuilt rollup
  of today's notes); included first so day-level abstractions land
  before per-event details.
* ``<daily_dir>/<today>/**/*.md`` — event notes for the day.

The diff vs ``file_catalog`` follows the same shape as
:class:`ScanCatalogChangesStep`: build ``existing`` (on-disk ``rel → mtime``)
and ``indexed`` (catalog ``rel → mtime``, restricted to today's
prefix so we never disturb entries from other days), then:

* ``existing`` keys not in ``indexed``      → **added**, dream
* mtime mismatch                              → **modified**, dream
* ``indexed`` keys not in ``existing``       → **deleted**, drop from catalog
* mtime match                                 → **unchanged**, skip

For every to-dream file, the step calls the configured ``dispatch_job``
(default ``"dream"``) via :meth:`BaseStep.run_job`. The dispatch job's
``Response`` carries a ``DreamResult`` payload in ``metadata``;
AutoDream re-hydrates that for its per-file aggregate.

After dreaming, successful (and Phase 1 vacuously-skipped) files
upsert their current ``st_mtime`` so the next tick re-dreams only
what actually changed. Failures leave the catalog untouched and will
be retried on the next tick.

The two writers (``WatchDreamStep`` event-driven, ``AutoDreamStep``
catch-up scan) share the protocol: upsert ``(path, st_mtime)`` on
success. Last-writer-wins is fine — same path + same content yields
the same mtime, so they cannot disagree on what's "done".

Cron scheduling itself is out of scope; this step is the unit of
work — invoke it from a system cron, ``reme auto-dream date=...``,
or any other catch-up trigger when ``auto_dream_loop`` missed a file
(e.g. process crashed before the watcher fired).

**Backend-agnostic.** Because dispatch goes through the configured
``dream`` job, the per-file dream implementation is decided by the
YAML config — whichever step the ``dream`` job's ``backend`` resolves
to. AutoDream itself doesn't care which backend runs underneath.

Inputs (RuntimeContext):
    date (str, optional): YYYY-MM-DD to scan. Defaults to today
        in the dreamer's timezone.
    hint (str, optional): passed through to each per-file dream.

Step kwargs (from yaml ``backend: auto_dream_step``):
    dispatch_job (str, default "dream"): name of the job to call
        per file. Override only if the deployment renamed the dream
        job. The dispatched job must accept ``path`` and ``hint``
        kwargs and return a ``Response`` whose ``metadata`` matches
        :class:`DreamResult`.
    persist (bool, default True): when True, ``file_catalog.dump()``
        is called after the batch so progress survives a restart.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from ._evolve import now
from .dream import DreamResult
from ..base_step import BaseStep
from ...components import R
from ...schema import FileNode


class AutoDreamResult(BaseModel):
    """Aggregated outcome of one auto-dream tick."""

    date: str = ""
    files_scanned: int = 0
    files_unchanged: int = 0
    files_dreamed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_deleted: int = 0
    per_file: list[DreamResult] = Field(default_factory=list)
    summary: str = ""


@R.register("auto_dream_step")
class AutoDreamStep(BaseStep):
    """Scan ``daily/<today>.md`` + ``daily/<today>/`` and dispatch to the
    configured ``dream`` job for each file whose ``st_mtime`` doesn't
    already match its ``file_catalog`` entry."""

    def __init__(self, dispatch_job: str = "dream", persist: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.dispatch_job: str = dispatch_job
        self.persist: bool = persist

    def _vault_dir(self) -> Path:
        """Vault root as an absolute path (mirrors :meth:`DreamStep._vault_dir`)."""
        vr = getattr(self.file_store, "vault_path", None)
        return Path(vr).resolve() if vr else Path.cwd().resolve()

    async def _dispatch_dream(self, rel_path: str, hint: str) -> DreamResult:
        """Call the configured ``dispatch_job`` once and re-hydrate its
        ``Response.metadata`` into a :class:`DreamResult`.

        On dispatch failure (job raises) returns a ``DreamResult`` with
        ``error`` populated so the caller's accounting stays uniform.
        """
        try:
            resp = await self.run_job(self.dispatch_job, path=rel_path, hint=hint)
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error(
                f"[{self.name}] dispatch {self.dispatch_job!r} failed on {rel_path}: " f"{type(e).__name__}: {e}",
            )
            return DreamResult(path=rel_path, error=f"{type(e).__name__}: {e}")

        # The dream job's execute() does context.response.metadata.update(result.model_dump()),
        # so metadata carries every DreamResult field. extra keys are
        # ignored by pydantic v2 default (extra='ignore').
        md = dict(resp.metadata or {})
        try:
            dr = DreamResult.model_validate(md)
        except Exception as e:  # noqa: BLE001
            self.logger.error(
                f"[{self.name}] dispatch {self.dispatch_job!r} returned non-DreamResult metadata "
                f"for {rel_path}: {type(e).__name__}: {e}",
            )
            return DreamResult(path=rel_path, error=f"bad dispatch metadata: {type(e).__name__}: {e}")

        # If the underlying step set success=False, treat as failure even
        # if metadata didn't carry an error string (defensive).
        if not resp.success and not dr.error:
            dr.error = resp.answer or "dispatch returned success=False"
        if not dr.path:
            dr.path = rel_path
        return dr

    async def execute(self):
        assert self.context is not None
        date_input: str = (self.context.get("date", "") or "").strip()
        hint: str = (self.context.get("hint", "") or "").strip()

        # daily_dir comes from app config — NOT a tool param. Same convention
        # as daily_create / daily_list / daily_reindex.
        cfg = self.app_context.app_config if self.app_context is not None else None
        daily_dir = (cfg.daily_dir if cfg else "") or "daily"

        tz = self.app_context.app_config.timezone if self.app_context is not None else None
        today = date_input or now(tz).strftime("%Y-%m-%d")
        vault = self._vault_dir()
        files = _scan_today_files(vault, today, daily_dir)

        # existing: today's on-disk paths → st_mtime. Insertion order = scan
        # order (date.md first, then sorted event notes); preserved through
        # the diff so the day-index file is dreamed before per-event notes.
        existing: dict[str, float] = {}
        for rel in files:
            try:
                existing[rel] = (vault / rel).stat().st_mtime
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {rel}: {e}")

        # indexed: catalog entries restricted to today's prefix. Restriction
        # is critical — get_nodes() returns all days, but we must only
        # consider deletions within today's scan scope.
        today_md = f"{daily_dir}/{today}.md"
        today_dir = f"{daily_dir}/{today}/"
        all_nodes = await self.file_catalog.get_nodes()
        indexed: dict[str, float] = {
            n.path: n.st_mtime for n in all_nodes if n.path == today_md or n.path.startswith(today_dir)
        }

        # Diff — same vocabulary as scan_*_changes_step (added/modified/deleted).
        to_dream: list[tuple[str, float]] = [(rel, mt) for rel, mt in existing.items() if indexed.get(rel) != mt]
        unchanged: list[str] = [rel for rel, mt in existing.items() if indexed.get(rel) == mt]
        to_delete: list[str] = sorted(indexed.keys() - existing.keys())

        result = AutoDreamResult(
            date=today,
            files_scanned=len(existing),
            files_unchanged=len(unchanged),
            files_deleted=len(to_delete),
        )
        self.logger.info(
            f"[{self.name}] auto-dream tick date={today} scanned={len(existing)} "
            f"unchanged={len(unchanged)} todo={len(to_dream)} deleted={len(to_delete)} "
            f"under {daily_dir}/{today}{{.md,/}} dispatch={self.dispatch_job!r}",
        )

        # Drop catalog entries for files no longer on disk first. Cheap, no
        # LLM, and keeps the catalog consistent even if the dream pass below
        # errors.
        if to_delete:
            try:
                await self.file_catalog.delete(to_delete)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.delete failed: {type(e).__name__}: {e}",
                )

        # Dispatch + upsert per-file. Single-file granularity means a job
        # failure on file N doesn't block files N+1..K from advancing their
        # catalog mtime. The dispatch decouples backend choice (AS / CC)
        # from this loop — the configured ``dream`` job picks the runner.
        upsert_nodes: list[FileNode] = []
        for rel_path, mtime in to_dream:
            dr = await self._dispatch_dream(rel_path, hint)
            result.per_file.append(dr)
            if dr.error:
                # Failures leave the catalog untouched — next tick retries.
                result.files_failed += 1
                continue
            if dr.skipped:
                # Phase 1 said "nothing to extract" — still mark as seen so
                # Phase 1 doesn't re-run on every tick.
                result.files_skipped += 1
            else:
                result.files_dreamed += 1
            upsert_nodes.append(FileNode(path=rel_path, st_mtime=mtime))

        if upsert_nodes:
            try:
                await self.file_catalog.upsert(upsert_nodes)
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.upsert failed: {type(e).__name__}: {e}",
                )

        if self.persist and (upsert_nodes or to_delete):
            try:
                await self.file_catalog.dump()
            except Exception as e:  # noqa: BLE001
                self.logger.exception(
                    f"[{self.name}] file_catalog.dump failed: {type(e).__name__}: {e}",
                )

        result.summary = _render_auto_dream_summary(result)
        self.context.response.success = result.files_failed == 0
        self.context.response.answer = result.summary
        self.context.response.metadata.update(result.model_dump())
        return self.context.response


def _scan_today_files(vault: Path, today: str, daily_dir: str) -> list[str]:
    """Return vault-relative paths of today's day-index + event notes.

    * ``<daily_dir>/<today>.md`` — the day-index file (auto-rebuilt
      rollup of all of today's notes). Included first so its day-level
      abstractions land before the per-event details.
    * ``<daily_dir>/<today>/**/*.md`` — event notes for the day,
      sorted by path for deterministic processing order.
    """
    out: list[str] = []
    if not daily_dir:
        return out

    day_index = vault / daily_dir / f"{today}.md"
    if day_index.is_file():
        out.append(str(day_index.relative_to(vault)))

    daily_root = vault / daily_dir / today
    if daily_root.is_dir():
        for md in sorted(daily_root.rglob("*.md")):
            if md.is_file():
                out.append(str(md.relative_to(vault)))

    return out


def _render_auto_dream_summary(r: AutoDreamResult) -> str:
    """One-line header + one line per dreamed file with its outcome.
    Unchanged + deleted files are counted in the header only."""
    lines = [
        f"[AutoDreamStep] date={r.date} scanned={r.files_scanned} "
        f"unchanged={r.files_unchanged} dreamed={r.files_dreamed} "
        f"skipped={r.files_skipped} failed={r.files_failed} "
        f"deleted={r.files_deleted}",
    ]
    for dr in r.per_file:
        if dr.error:
            status = f"ERROR ({dr.error})"
        elif dr.skipped:
            status = "SKIP"
        else:
            status = f"OK (+{len(dr.nodes_created)} created, ~{len(dr.nodes_updated)} updated)"
        lines.append(f"  - {dr.path}: {status}")
    return "\n".join(lines)
