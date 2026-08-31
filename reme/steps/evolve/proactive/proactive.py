"""Read interests.yaml for proactive consumption (F5).

Migrated from ``dream/proactive.py`` (INV-9 mechanical migration) and extended
with ``min_confidence`` filtering (default 0.4, safely below the 0.5 confidence
fallback) and ``horizon_days``: horizon=1 keeps the legacy single-day file
read (v1 compatible); horizon>1 reads the truth source directly, filtering by
``last_evidence_at`` recency (v5 R4 - the truth source already carries the
cross-day merge, so the reader no longer re-scans exposure products).
"""

import datetime as dt

import yaml

from ...base_step import BaseStep
from ....components import R
from ....schema import ProactiveResult
from ..dream.utils import load_yaml_topics, today, workspace_dir
from .utils import (
    dump_topic,
    load_state,
    parse_interests_topics,
    quarantine_interests,
    sort_topics,
    topic_id,
)


@R.register("proactive_step")
class ProactiveStep(BaseStep):
    """Read ``daily/<date>/interests.yaml`` (optionally merged over a horizon).

    - v1 files (nightly) are read as ``push=true`` and never rewritten;
    - ``push=false`` days contribute nothing;
    - resolved ids (truth source registry) are suppressed;
    - ``horizon_days>1`` reads the truth source and filters by evidence recency.
    """

    def __init__(self, include_content: bool = True, horizon_days: int = 1, min_confidence: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.include_content = include_content
        self.horizon_days = max(int(horizon_days), 1)
        self.min_confidence = float(min_confidence)

    async def execute(self):
        assert self.context is not None
        day = today(self, str(self.context.get("date", "") or ""))
        include_content = bool(self.context.get("include_content", self.include_content))
        horizon = int(self.context.get("horizon_days", self.horizon_days) or self.horizon_days)
        horizon = max(horizon, 1)
        raw_min_confidence = self.context.get("min_confidence", self.min_confidence)
        try:
            min_confidence = float(raw_min_confidence)
        except (TypeError, ValueError):
            min_confidence = self.min_confidence
        daily = self.config_value("daily_dir")
        ws = workspace_dir(self)
        default_rel = f"{daily}/{day}/interests.yaml"
        result = ProactiveResult(date=day, path=default_rel)
        self.logger.info(
            f"[{self.name}] start date={day} path={default_rel} include_content={include_content} "
            f"horizon_days={horizon} min_confidence={min_confidence}",
        )

        if horizon == 1:
            outcome = self._read_single(ws, daily, day, include_content, min_confidence, result)
            if outcome is not None:
                return outcome

        return self._read_horizon(ws, daily, day, horizon, include_content, min_confidence, result)

    # ------------------------------------------------------------------
    # Single-day read (legacy-compatible path)
    # ------------------------------------------------------------------

    def _read_single(self, ws, daily, day, include_content, min_confidence, result: ProactiveResult):
        rel_path = f"{daily}/{day}/interests.yaml"
        abs_path = ws / daily / day / "interests.yaml"
        if not abs_path.is_file():
            result.skipped, result.summary = True, f"Skipped: interests file not found at {rel_path}"
            self.logger.info(f"[{self.name}] skip missing path={rel_path}")
            return self._finish(True, result, include_content=include_content)
        try:
            raw_text = abs_path.read_text(encoding="utf-8")
            try:
                data = yaml.safe_load(raw_text)
                if not isinstance(data, dict):
                    raise ValueError("interests.yaml is not a mapping")
            except Exception as e:  # noqa: BLE001
                quarantine_interests(abs_path, e)
                data = {}  # A2: treat corrupt file as empty
            topics, is_v1, push = parse_interests_topics(data, day)
            if push is False:
                result.skipped = True
                result.summary = f"Skipped: interests file at {rel_path} has push=false"
                result.push = False
                self.logger.info(f"[{self.name}] skip push=false path={rel_path}")
                return self._finish(True, result, include_content=include_content)

            result.push = True
            result.content = raw_text if include_content else ""
            if is_v1:
                # Legacy shape (title/reason/evidence/keywords/paths) for v1 files;
                # v1 topics carry no confidence and fall back to 0.5 for filtering.
                resolved_ids = self._resolved_ids(ws, daily)
                legacy_topics = load_yaml_topics(abs_path)
                kept = [t for t in legacy_topics if topic_id(str(t.get("title") or "")) not in resolved_ids]
                if min_confidence > 0.5 + 1e-9:  # v1 topics carry no confidence; fallback is 0.5 (F1.1)
                    kept = []
                result.topics = kept
            else:
                result.generated_at = str(data.get("generated_at") or "")
                resolved_ids = self._resolved_ids(ws, daily)
                kept = [
                    dump_topic(t)
                    for t in sort_topics(topics)
                    if t.id not in resolved_ids and t.confidence >= min_confidence - 1e-9
                ]
                result.topics = kept
                agenda_raw = data.get("agenda")
                if isinstance(agenda_raw, list):
                    kept_ids = {str(t.get("id") or "") for t in kept}
                    result.agenda = [
                        item
                        for item in agenda_raw
                        if isinstance(item, dict) and str(item.get("topic_id") or "") in kept_ids
                    ]
            result.summary = f"Read {len(result.topics)} proactive topic(s) from {rel_path}"
            self.logger.info(f"[{self.name}] read done path={rel_path} topics={len(result.topics)}")
            return self._finish(True, result, include_content=include_content)
        except Exception as e:  # noqa: BLE001
            result.error, result.summary = f"{type(e).__name__}: {e}", ""
            self.logger.error(f"[{self.name}] read failed path={rel_path}: {result.error}")
            return self._finish(False, result, include_content=include_content)

    # ------------------------------------------------------------------
    # Multi-day merge (horizon_days > 1, or fallback target)
    # ------------------------------------------------------------------

    def _read_horizon(self, ws, daily, day, horizon, include_content, min_confidence, result: ProactiveResult):
        """Truth-source view (v5 R4): open topics with evidence inside the horizon.

        The truth source already carries the cross-day merge (carry-forward),
        so the reader no longer re-scans N days of exposure products.
        """
        state_file, _needs_bootstrap = load_state(ws, daily)
        if not state_file.open_topics:
            result.skipped = True
            result.summary = f"Skipped: truth source has no open topics (horizon_days={horizon})"
            self.logger.info(f"[{self.name}] skip empty truth source horizon={horizon}")
            return self._finish(True, result, include_content=include_content)
        try:
            base = dt.date.fromisoformat(day)
            cutoff = (base - dt.timedelta(days=max(horizon - 1, 0))).isoformat()
        except ValueError:
            cutoff = ""
        kept = [
            dump_topic(t)
            for t in state_file.open_topics
            if (not cutoff or str(t.last_evidence_at or "") >= cutoff) and t.confidence >= min_confidence - 1e-9
        ]
        kept = sort_topics(kept)
        result.path = f"{daily}/_proactive.yaml"
        result.topics = kept
        result.summary = f"Read {len(kept)} proactive topic(s) from the truth source (horizon_days={horizon})"
        self.logger.info(f"[{self.name}] truth-source read done horizon={horizon} topics={len(kept)}")
        return self._finish(True, result, include_content=include_content)

    def _resolved_ids(self, ws, daily: str) -> set[str]:
        state_file, _needs_bootstrap = load_state(ws, daily)
        return {str(r.get("id") or "") for r in state_file.resolved if isinstance(r, dict) and r.get("id")}

    def _finish(self, success: bool, result: ProactiveResult, *, include_content: bool):
        assert self.context is not None
        self.context.response.success = success
        if not success:
            self.context.response.answer = f"Error: {result.error}"
        elif result.skipped:
            self.context.response.answer = result.summary
        else:
            self.context.response.answer = {
                "summary": result.summary,
                "topics": result.topics,
                **({"agenda": result.agenda} if result.agenda else {}),
                **({"content": result.content} if include_content else {}),
            }
        self.context.response.metadata.update(result.model_dump())
        self.logger.info(f"[{self.name}] finish success={success} answer={self.context.response.answer!r}")
        return self.context.response
