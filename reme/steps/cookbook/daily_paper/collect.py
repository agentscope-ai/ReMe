"""Collect ranked Hugging Face papers for the daily-paper workflow."""

import asyncio
import datetime as dt
from pathlib import Path

import frontmatter

from ....components import R
from ....schema import PaperInfo, PaperSelection
from ....utils.arxiv import ARXIV_ID_PATTERN
from ....utils.huggingface_papers import HuggingFacePapersClient
from ...evolve import now
from ._common import DailyPaperStep


@R.register("daily_paper_collect_step")
class DailyPaperCollectStep(DailyPaperStep):
    """Collect current weekly/monthly rankings and strict-yesterday exclusions."""

    @staticmethod
    def _strict_date(value: str) -> dt.date:
        text = str(value or "").strip()
        try:
            parsed = dt.date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("date must be YYYY-MM-DD") from exc
        if parsed.isoformat() != text:
            raise ValueError("date must be YYYY-MM-DD")
        return parsed

    @staticmethod
    def _paper_scope_values(day: dt.date) -> tuple[str, str]:
        iso = day.isocalendar()
        return f"{iso.year}-W{iso.week:02d}", day.strftime("%Y-%m")

    @staticmethod
    def load_historical_arxiv_ids(
        workspace: Path,
        run_date: dt.date,
        history_days: int,
        daily_dir: str,
    ) -> set[str]:
        """Read previously recommended paper ids from prior daily note frontmatter."""
        if history_days <= 0:
            return set()
        earliest, root = run_date - dt.timedelta(days=history_days), workspace / daily_dir
        if not root.is_dir():
            return set()

        found: set[str] = set()
        for day_dir in root.iterdir():
            if not day_dir.is_dir():
                continue
            try:
                note_date = dt.date.fromisoformat(day_dir.name)
            except ValueError:
                continue
            if not earliest <= note_date < run_date:
                continue
            for note_path in day_dir.glob("paper-*.md"):
                try:
                    metadata = frontmatter.load(note_path).metadata
                except (OSError, UnicodeError, ValueError):
                    continue
                arxiv_id = str(metadata.get("arxiv_id") or "").strip()
                if ARXIV_ID_PATTERN.fullmatch(arxiv_id):
                    found.add(arxiv_id)
        return found

    @staticmethod
    def load_saved_selection(digest_path: Path) -> dict | None:
        """Rebuild the saved selection from the digest and paper-note frontmatter."""
        try:
            digest_metadata = frontmatter.load(digest_path).metadata
        except (OSError, UnicodeError, ValueError):
            return None
        arxiv_ids = digest_metadata.get("arxiv_ids")
        if not isinstance(arxiv_ids, list) or not arxiv_ids:
            return None

        selected = []
        for rank, value in enumerate(arxiv_ids, start=1):
            arxiv_id = str(value or "").strip()
            if not ARXIV_ID_PATTERN.fullmatch(arxiv_id):
                return None
            try:
                note_metadata = frontmatter.load(digest_path.parent / f"paper-{arxiv_id}.md").metadata
            except (OSError, UnicodeError, ValueError):
                return None
            selected.append(
                {
                    "arxiv_id": arxiv_id,
                    "rank": rank,
                    "reason": str(note_metadata.get("selection_reason") or "").strip(),
                    "memory_relevance": note_metadata.get("memory_relevance"),
                },
            )

        try:
            selection = PaperSelection.model_validate(
                {
                    "selection_reasoning": str(digest_metadata.get("selection_reasoning") or "").strip(),
                    "selected": selected,
                    "alternates": digest_metadata.get("alternate_arxiv_ids") or [],
                },
            )
        except (TypeError, ValueError):
            return None
        return selection.model_dump()

    @staticmethod
    def _merge_paper(existing: PaperInfo | None, incoming: PaperInfo) -> PaperInfo:
        if existing is None:
            return incoming.model_copy(deep=True)
        values = existing.model_dump()
        for key, value in incoming.model_dump().items():
            if key == "upvotes":
                values[key] = max(int(values[key] or 0), int(value or 0))
            elif values.get(key) in (None, "", []):
                values[key] = value
        return PaperInfo.model_validate(values)

    async def execute(self):
        assert self.context is not None
        timezone = self.app_context.app_config.timezone if self.app_context is not None else None
        raw_date = str(self._value("date", "") or "").strip()
        run_date = self._strict_date(raw_date) if raw_date else now(timezone).date()
        day = run_date.isoformat()
        self._set_state("run_date", day)

        daily_dir = str(self.config_value("daily_dir")).strip("/")
        digest_rel = f"{daily_dir}/{day}/daily-paper-brief.md"
        force = bool(self._value("force", False))
        self.logger.info(f"[{self.name}] start date={day} force={force}")
        digest_path = self.workspace_path / digest_rel
        if digest_path.is_file() and not force:
            self._set_state("skip", True)
            self._set_state("digest_path", digest_rel)
            self.context.response.success = True
            self.context.response.answer = f"Skipped: daily paper brief already exists at {digest_rel}"
            self.context.response.metadata.update({"date": day, "digest_path": digest_rel, "skipped": True})
            if selection := self.load_saved_selection(digest_path):
                self.context.response.metadata["selection"] = selection
            self.logger.info(f"[{self.name}] skip existing digest path={digest_rel}")
            return self.context.response

        week, month = self._paper_scope_values(run_date)
        yesterday = (run_date - dt.timedelta(days=1)).isoformat()
        self.logger.info(
            f"[{self.name}] fetch start week={week} month={month} yesterday={yesterday}",
        )
        async with HuggingFacePapersClient(
            timeout=float(self._value("hf_timeout", 30.0)),
            max_retries=int(self._value("hf_max_retries", 3)),
        ) as client:
            weekly, monthly, yesterday_ids = await asyncio.gather(
                client.fetch_scope("week", week),
                client.fetch_scope("month", month),
                client.fetch_daily_ids(yesterday),
            )
        self.logger.info(
            f"[{self.name}] fetch done weekly={len(weekly)} monthly={len(monthly)} " f"yesterday={len(yesterday_ids)}",
        )

        merged: dict[str, PaperInfo] = {}
        for rank, paper in enumerate(monthly, start=1):
            merged[paper.arxiv_id] = self._merge_paper(merged.get(paper.arxiv_id), paper)
            merged[paper.arxiv_id].monthly_rank = rank
        for rank, paper in enumerate(weekly, start=1):
            merged[paper.arxiv_id] = self._merge_paper(merged.get(paper.arxiv_id), paper)
            merged[paper.arxiv_id].weekly_rank = rank

        historical_ids = self.load_historical_arxiv_ids(
            self.workspace_path,
            run_date,
            int(self._value("history_days", 30)),
            daily_dir,
        )
        eligible = {key: paper for key, paper in merged.items() if key not in yesterday_ids | historical_ids}
        self.logger.info(
            f"[{self.name}] filter done merged={len(merged)} excluded_yesterday={len(yesterday_ids)} "
            f"excluded_history={len(historical_ids)} eligible={len(eligible)}",
        )
        if not eligible:
            raise RuntimeError("No eligible papers remain after yesterday and history exclusions")

        for key, value in {
            "info": eligible,
            "week": week,
            "month": month,
            "yesterday": yesterday,
            "excluded_yesterday": sorted(yesterday_ids),
            "excluded_history": sorted(historical_ids),
            "source_counts": {"weekly": len(weekly), "monthly": len(monthly), "merged": len(merged)},
        }.items():
            self._set_state(key, value)
        self.context.response.success = True
        self.context.response.answer = f"Collected {len(eligible)} eligible papers"
        self.logger.info(f"[{self.name}] finish date={day} eligible={len(eligible)}")
        return self.context.response
