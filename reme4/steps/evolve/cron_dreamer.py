"""CronDreamer — daily-tick wrapper around :class:`Dreamer`.

Inherits the per-file pipeline from :meth:`Dreamer.dream_one` and
adds the outer loop over today's daily/ + resource/ files. Cron
scheduling itself is out of scope; this step is just the unit of
work — invoke it from a system cron, ``reme auto-dream date=...``,
or any other catch-up trigger when ``auto_dream_loop`` missed a
file (e.g. process crashed before the watcher fired).

Inputs (RuntimeContext):
    date (str, optional): YYYY-MM-DD to scan. Defaults to today
        in the dreamer's timezone.
    hint (str, optional): passed through to each per-file dream.

The outer loop reuses ``Dreamer``'s prompt mounting via MRO — no
separate YAML; ``auto_dream.yaml`` is found through the parent class.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from .auto_dream import Dreamer, DreamResult
from ...components import R


class CronDreamResult(BaseModel):
    """Aggregated outcome of one cron tick."""

    date: str = ""
    files_scanned: int = 0
    files_dreamed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    per_file: list[DreamResult] = Field(default_factory=list)
    summary: str = ""


@R.register("cron_dreamer_step")
class CronDreamer(Dreamer):
    """Loop ``daily/<today>/`` + ``resource/<today>/`` and dream each file."""

    async def execute(self):
        assert self.context is not None
        date_input: str = (self.context.get("date", "") or "").strip()
        hint: str = (self.context.get("hint", "") or "").strip()

        # daily_dir / resource_dir come from app config — NOT tool params.
        # Same convention as daily_create / daily_list / daily_reindex.
        # resource_dir may be empty (default) — that just skips the resource scan.
        cfg = self.app_context.app_config if self.app_context is not None else None
        daily_dir = (cfg.daily_dir if cfg else "") or "daily"
        resource_dir = cfg.resource_dir if cfg else ""

        today = date_input or self._now().strftime("%Y-%m-%d")
        vault = self._vault_dir()
        files = _scan_today_files(vault, today, daily_dir, resource_dir)

        result = CronDreamResult(date=today, files_scanned=len(files))
        self.logger.info(
            f"[{self.name}] cron tick date={today} scanned={len(files)} file(s) under "
            f"{daily_dir}/{today}/ + {resource_dir}/{today}/",
        )

        for rel_path in files:
            try:
                dr = await self.dream_one(rel_path, hint)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.error(
                    f"[{self.name}] dream_one failed on {rel_path}: {type(e).__name__}: {e}",
                )
                dr = DreamResult(
                    path=rel_path,
                    error=f"{type(e).__name__}: {e}",
                )
            result.per_file.append(dr)
            if dr.error:
                result.files_failed += 1
            elif dr.skipped:
                result.files_skipped += 1
            else:
                result.files_dreamed += 1

        result.summary = _render_cron_summary(result)
        self.context.response.success = result.files_failed == 0
        self.context.response.answer = result.summary
        self.context.response.metadata.update(result.model_dump())


def _scan_today_files(
    vault: Path,
    today: str,
    daily_dir: str,
    resource_dir: str,
) -> list[str]:
    """Return vault-relative paths of today's daily notes + resource files.

    * ``<daily_dir>/<today>.md`` — the day-index file (auto-rebuilt
      rollup of all of today's notes). Included first so its day-level
      abstractions land before the per-event details.
    * ``<daily_dir>/<today>/**/*.md`` — event notes for the day,
      sorted by path.
    * ``<resource_dir>/<today>/**/*`` — any file type ingested under
      today's resource folder. Skipped when ``resource_dir`` is empty.

    Results are sorted for deterministic processing order within each
    group; the day-index file leads.
    """
    out: list[str] = []

    if daily_dir:
        day_index = vault / daily_dir / f"{today}.md"
        if day_index.is_file():
            out.append(str(day_index.relative_to(vault)))
        daily_root = vault / daily_dir / today
        if daily_root.is_dir():
            for md in sorted(daily_root.rglob("*.md")):
                if md.is_file():
                    out.append(str(md.relative_to(vault)))

    if resource_dir:
        resource_root = vault / resource_dir / today
        if resource_root.is_dir():
            for f in sorted(p for p in resource_root.rglob("*") if p.is_file()):
                out.append(str(f.relative_to(vault)))

    return out


def _render_cron_summary(r: CronDreamResult) -> str:
    """One-line header + one line per file with its outcome."""
    lines = [
        f"[CronDreamer] date={r.date} scanned={r.files_scanned} "
        f"dreamed={r.files_dreamed} skipped={r.files_skipped} failed={r.files_failed}",
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
