"""Proactive topics step: dedup, truth-source update, derived push, render (F2.4).

Pure computation step (v5 R2): no LLM calls, no budget, no agent_wrapper.
Semantic dedup keeps only the ``>= known_threshold`` "known" drop; everything
below is kept (loose-not-leaky). The main dedup defense lives upstream in the
extract prompt's same-matter rule plus same-id merging here.

``known_threshold`` is bound to the embedding model, dimensions AND vector-text
scheme it was calibrated for (v5.2: ``title。reason`` texts, threshold 0.85,
34-pair measurement: DUP band 0.773-0.943 vs KEEP band <=0.772). Cosine
magnitudes are not comparable across models, so a fingerprint mismatch degrades
the gate to exact normalize comparison instead of silently misfiring. Without
any configured embedder the step skips the semantic gate entirely and the
workflow continues on exact comparison (BM25-only deployments).
"""

import datetime as dt
import re
import time

from ...base_step import BaseStep
from .._evolve import passthrough_response
from ....components import R
from ....enumeration import ComponentEnum
from ....schema import ProactiveState, ProactiveStateFile, ProactiveTopic
from ....schema.proactive import clamp_confidence
from ..dream.utils import previous_dates, today, workspace_dir
from .utils import (
    current_now,
    dump_topic,
    interests_path_for,
    load_state,
    norm_path,
    normalize_topic,
    parse_interests_topics,
    read_interests_data,
    render_interests,
    save_state,
    sort_topics,
    trim_state_file,
    write_interests_if_changed,
)


@R.register("proactive_topics_step")
class ProactiveTopicsStep(BaseStep):
    """Filter candidates, update the truth source, and render interests.yaml.

    Without ``as_embedding`` the exact ``normalize_topic`` comparison applies.
    With embeddings, ``sim >= known_threshold`` drops as known; below the
    threshold candidates are kept. Candidates whose id matches a resolved
    tombstone are resurrected (tombstone removed, original ``first_seen``
    kept) instead of being silently suppressed.
    """

    def __init__(
        self,
        known_threshold: float = 0.85,
        min_push_confidence: float = 0.5,
        max_topics: int = 10,
        dedup_lookback_days: int = 7,
        digest_compare_limit: int = 500,
        known_threshold_calibrated_for: str = "text-embedding-v4@1024",
        skip_key: str = "proactive_skip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.known_threshold = float(known_threshold)
        self.known_threshold_calibrated_for = str(known_threshold_calibrated_for or "")
        self.min_push_confidence = float(min_push_confidence)
        self.max_topics = max(int(max_topics), 1)
        self.dedup_lookback_days = max(int(dedup_lookback_days), 0)
        self.digest_compare_limit = max(int(digest_compare_limit), 0)
        self.skip_key = skip_key

    # pylint: disable=too-many-statements
    async def execute(self):
        assert self.context is not None
        if self.context.get(self.skip_key):
            return passthrough_response(self, self.skip_key)
        started = time.monotonic()
        if not self.context.get("proactive"):
            self.context.response.success = True
            self.context.response.answer = "Skipped topics: no proactive extract state in context"
            return self.context.response
        state = ProactiveState.model_validate(self.context.get("proactive"))
        if state.early_exit:
            self.context.response.success = True
            self.context.response.answer = f"Skipped topics: {state.early_exit}"
            return self.context.response
        day = state.date or today(self, str(self.context.get("date", "") or ""))
        ws = workspace_dir(self)
        daily = state.daily_dir or "daily"
        self.logger.info(
            f"[{self.name}] start date={day} follow_ups={len(state.follow_ups)} extends={len(state.extends)} "
            f"updates={len(state.updates)} carry_forward={len(state.carry_forward_all)}",
        )

        state_file, _needs_bootstrap = load_state(ws, daily)
        open_by_id = {t.id: t for t in state_file.open_topics if t.id}
        expiry_cutoff = self._expiry_cutoff(day, max(state.carry_forward_days, 1))

        # 1) Apply updates to the truth source (F2.3). An action=update is only
        # legal when its evidence anchors this round's new material (v5.2 hard
        # check, same shape as INV-8); otherwise it degrades to keep so stale
        # re-reads cannot rewrite evidence or refresh the freshness sort key.
        new_material = {norm_path(str(path).split("#", 1)[0]) for path in state.changed_paths}
        for update in state.updates:
            topic = open_by_id.get(str(update.get("id") or ""))
            if topic is None:
                continue
            action = str(update.get("action") or "keep")
            if action == "update":
                evidence_raw = str(update.get("evidence") or "").strip()
                evidence_path = norm_path(evidence_raw.split("#", 1)[0])
                if evidence_path in new_material:
                    topic.last_evidence_at = self._evidence_date(evidence_path) or day
                    topic.evidence = evidence_raw[:120]
                    if update.get("confidence") is not None:
                        topic.confidence = clamp_confidence(update.get("confidence"))
                    state.updates_applied += 1
                else:
                    self.logger.info(
                        f"[{self.name}] update for {topic.id} downgraded to keep: evidence "
                        f"{evidence_path or '<empty>'!r} not in this round's new material",
                    )
            elif action == "resolve":
                state_file.open_topics = [t for t in state_file.open_topics if t.id != topic.id]
                del open_by_id[topic.id]
                state_file.resolved.append(
                    {
                        "id": topic.id,
                        "title": topic.title,
                        "resolved_at": day,
                        "first_seen": topic.first_seen,
                        "evidence": str(update.get("evidence") or "")[:120],
                    },
                )
                state.updates_resolved += 1

        # 2) Candidates: same-id merge, tombstone resurrect, dedup the rest.
        raw_candidates = list(state.follow_ups) + list(state.extends)
        state.candidates_in = len(raw_candidates)
        merged: list[dict] = []
        resurrected: list[dict] = []
        fresh: list[dict] = []
        tombstone_by_id = {
            str(r.get("id") or ""): r for r in state_file.resolved if isinstance(r, dict) and r.get("id")
        }
        for candidate in raw_candidates:
            cid = str(candidate.get("id") or "")
            existing = open_by_id.get(cid)
            if existing is not None:
                if expiry_cutoff and str(existing.first_seen or "") <= expiry_cutoff:
                    # Over-age re-mention: trim would prune it this round, so a
                    # fresh mention restarts the lifetime (unfinished business
                    # must keep being re-executed, v5.1).
                    self.logger.info(
                        f"[{self.name}] over-age topic {cid} re-mentioned; restarting first_seen "
                        f"({existing.first_seen} -> {day})",
                    )
                    existing.first_seen = day
                self._merge_into(existing, candidate, day)
                merged.append(candidate)
                continue
            tombstone = tombstone_by_id.get(cid)
            if tombstone is not None:
                topic = self._resurrect(tombstone, candidate, day)
                state_file.resolved = [r for r in state_file.resolved if r is not tombstone]
                del tombstone_by_id[cid]
                state_file.open_topics.append(topic)
                open_by_id[cid] = topic
                resurrected.append(candidate)
                self.logger.info(f"[{self.name}] resurrected resolved topic {cid} ({topic.title!r})")
                continue
            fresh.append(candidate)

        dropped_duplicate = dropped_known = 0
        survivors: list[dict] = []
        seen_normalized: set[str] = set()
        if fresh:
            comparison = await self._comparison_texts(ws, daily, day, state_file, expiry_cutoff)
            embedder = self._resolve_embedding()
            if embedder is not None and self.known_threshold_calibrated_for:
                fingerprint = self._embedding_fingerprint(embedder)
                if fingerprint != self.known_threshold_calibrated_for:
                    self.logger.warning(
                        f"[{self.name}] embedding fingerprint {fingerprint or '<unknown>'!r} != "
                        f"known_threshold calibration {self.known_threshold_calibrated_for!r}; "
                        f"cosine is not comparable across models, degrading to exact normalize comparison",
                    )
                    embedder = None
            comparison_norms = {normalize_topic(title) for title, _ in comparison}
            for candidate in fresh:
                normalized = normalize_topic(str(candidate.get("title") or ""))
                if normalized in seen_normalized:
                    dropped_duplicate += 1
                    continue
                if embedder is None:
                    if normalized in comparison_norms:
                        dropped_duplicate += 1
                        continue
                else:
                    verdict, matched, similarity = await self._semantic_verdict(
                        embedder,
                        candidate,
                        comparison,
                        comparison_norms,
                    )
                    if verdict == "known":
                        dropped_known += 1
                        self.logger.info(
                            f"[{self.name}] candidate {candidate.get('title')!r} dropped as known "
                            f"(sim={similarity:.3f}, matched={matched!r})",
                        )
                        continue
                    if verdict == "duplicate":  # embedding failure fallback
                        dropped_duplicate += 1
                        continue
                seen_normalized.add(normalized)
                survivors.append(candidate)
        state.dropped_duplicate = dropped_duplicate
        state.dropped_known = dropped_known
        state.candidates = merged + resurrected + survivors
        self.logger.info(
            f"[{self.name}] candidates in={state.candidates_in} merged={len(merged)} "
            f"resurrected={len(resurrected)} new={len(survivors)} "
            f"dropped_duplicate={dropped_duplicate} dropped_known={dropped_known}",
        )

        # 3) Truth source: add new topics, prune, single atomic write (F1.3).
        for candidate in survivors:
            state_file.open_topics.append(ProactiveTopic.model_validate(candidate))
        trim_state_file(state_file, day, max(state.carry_forward_days, 1))
        await save_state(ws, state_file, daily)

        # 4) Push derived from the cumulative truth source (v5 R1): a topic
        # discovered today with sufficient confidence. Monotonic across same-day
        # rounds because such topics persist in the truth source once added.
        push = any(t.first_seen == day and t.confidence >= self.min_push_confidence for t in state_file.open_topics)
        if push:
            file_skip_reason = ""
        elif merged or resurrected or survivors:
            file_skip_reason = "low_confidence"
        else:
            file_skip_reason = "all_duplicates"

        # 5) Render from the truth source and write idempotently (A4).
        topics_out = sort_topics(list(state_file.open_topics))[: self.max_topics]
        now_dt = current_now(self)
        rendered = render_interests(day, topics_out, push, now_dt)
        interests_path = interests_path_for(ws, daily, day)
        written = write_interests_if_changed(ws, interests_path, rendered)
        rel_path = norm_path(interests_path.relative_to(ws).as_posix())

        state.topics_out = [dump_topic(t) for t in topics_out]
        state.push = push
        state.file_skip_reason = file_skip_reason
        state.interests_path = rel_path
        state.interests_written = written
        state.duration_ms = int((time.monotonic() - started) * 1000)
        self._store(state)
        answer = (
            f"Topics: {len(topics_out)} rendered, push={push}, "
            f"skip_reason={file_skip_reason or '-'}, written={written} to {rel_path}"
        )
        self.context.response.success = True
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish {answer}")
        return self.context.response

    @staticmethod
    def _merge_into(existing: ProactiveTopic, candidate: dict, day: str) -> None:
        """Same-id candidate refreshes evidence in place; first_seen is kept."""
        existing.last_evidence_at = day
        if candidate.get("reason"):
            existing.reason = str(candidate["reason"])
        if candidate.get("evidence"):
            existing.evidence = str(candidate["evidence"])[:120]
        existing.confidence = clamp_confidence(candidate.get("confidence"))
        if candidate.get("keywords"):
            existing.keywords = list(candidate["keywords"])
        if candidate.get("paths"):
            existing.paths = list(candidate["paths"])

    @staticmethod
    def _resurrect(tombstone: dict, candidate: dict, day: str) -> ProactiveTopic:
        """Reopen a resolved topic: original first_seen kept, evidence refreshed.

        The over-age trim still applies to the original ``first_seen``, so a
        resurrection only extends a lifetime that has not fully elapsed.
        """
        return ProactiveTopic(
            id=str(tombstone.get("id") or candidate.get("id") or ""),
            title=str(candidate.get("title") or tombstone.get("title") or ""),
            reason=str(candidate.get("reason") or ""),
            kind=str(candidate.get("kind") or "interest_extend"),
            confidence=clamp_confidence(candidate.get("confidence")),
            first_seen=str(tombstone.get("first_seen") or day),
            last_evidence_at=day,
            evidence=str(candidate.get("evidence") or "")[:120],
            keywords=candidate.get("keywords") or [],
            paths=candidate.get("paths") or [],
        )

    async def _comparison_texts(
        self,
        ws,
        daily: str,
        day: str,
        state_file: ProactiveStateFile,
        expiry_cutoff: str = "",
    ) -> list[tuple[str, str]]:
        """(title, embed_text) pairs from recent interests, open topics, digest nodes.

        Embed text carries the reason when available (v5.2 calibration:
        title+reason separates the DUP/KEEP bands; bare titles do not).
        Over-age open topics are excluded (v5.1): they are pruned by trim this
        round, so using them as "known" would silently swallow a re-mention of
        a matter that is about to restart.
        """
        pairs: list[tuple[str, str]] = []
        for previous_day in previous_dates(day, self.dedup_lookback_days):
            data = read_interests_data(interests_path_for(ws, daily, previous_day))
            if not data:
                continue
            topics, _is_v1, _push = parse_interests_topics(data, previous_day)
            pairs.extend((t.title, _known_text(t.title, getattr(t, "reason", ""))) for t in topics)
        pairs.extend(
            (t.title, _known_text(t.title, t.reason))
            for t in state_file.open_topics
            if not expiry_cutoff or str(t.first_seen or "") > expiry_cutoff
        )
        pairs.extend((title, title) for title in await self._digest_titles())
        return pairs

    async def _digest_titles(self) -> list[str]:
        if self.app_context is None or self.digest_compare_limit <= 0:
            return []
        catalog = self.app_context.components.get(ComponentEnum.FILE_CATALOG, {}).get("digest")
        if catalog is None:
            return []
        try:
            nodes = await catalog.get_nodes()
        except Exception:  # noqa: BLE001
            return []
        ordered = sorted(nodes, key=lambda n: float(getattr(n, "st_mtime", 0.0) or 0.0), reverse=True)
        return [str(n.path).rsplit("/", 1)[-1].rsplit(".", 1)[0] for n in ordered[: self.digest_compare_limit]]

    def _resolve_embedding(self):
        if self.context is not None:
            candidate = self.context.get("as_embedding")
            if candidate is not None:
                return candidate
        name = self.kwargs.get("as_embedding", "default")
        if self.app_context is None:
            return None
        return self.app_context.components.get(ComponentEnum.AS_EMBEDDING, {}).get(name)

    async def _semantic_verdict(
        self,
        embedder,
        candidate: dict,
        comparison: list[tuple[str, str]],
        comparison_norms: set[str],
    ) -> tuple[str, str, float]:
        """Semantic gate: known (>= known_threshold) | keep; duplicate on embedding failure."""
        candidate_text = _embed_text(candidate)
        texts = [candidate_text] + [embed_text for _, embed_text in comparison]
        try:
            vectors = await embedder(texts)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"[{self.name}] embedding failed, falling back to exact dedup: {e}")
            normalized = normalize_topic(str(candidate.get("title") or ""))
            return ("duplicate" if normalized in comparison_norms else "keep"), "", 0.0
        if not vectors or len(vectors) != len(texts):
            return "keep", "", 0.0
        best, best_idx = 0.0, -1
        for idx, other in enumerate(vectors[1:]):
            similarity = _cosine(vectors[0], other)
            if similarity > best:
                best, best_idx = similarity, idx
        matched = comparison[best_idx][0] if 0 <= best_idx < len(comparison) else ""
        if best >= self.known_threshold:
            return "known", matched, best
        return "keep", matched, best

    @staticmethod
    def _expiry_cutoff(day: str, carry_forward_days: int) -> str:
        """ISO cutoff matching trim_state_file: first_seen <= cutoff is over-age."""
        try:
            base = dt.date.fromisoformat(day)
        except ValueError:
            return ""
        return (base - dt.timedelta(days=max(int(carry_forward_days), 0))).isoformat()

    @staticmethod
    def _evidence_date(path: str) -> str:
        """Date embedded in a daily evidence path; '' when unparseable (v5.2).

        Lets last_evidence_at reflect when the evidence actually happened
        (daily/2026-08-12/x.md -> 2026-08-12) instead of always today, so an
        update anchored on an older file cannot game the freshness sort key.
        """
        match = _EVIDENCE_DATE_RE.search(path or "")
        if match:
            try:
                return dt.date.fromisoformat(match.group(1)).isoformat()
            except ValueError:
                return ""
        return ""

    @staticmethod
    def _embedding_fingerprint(embedder) -> str:
        """`model@dimensions` of the resolved embedder; '' when not introspectable."""
        model = ""
        kwargs = getattr(embedder, "kwargs", None)
        if isinstance(kwargs, dict):
            model = str(kwargs.get("model") or "")
        if not model:
            model = str(getattr(getattr(embedder, "model", None), "model", "") or "")
        try:
            dimensions = int(getattr(embedder, "dimensions", 0) or 0)
        except Exception:  # noqa: BLE001 - property may raise RuntimeError pre-init
            dimensions = 0
        return f"{model}@{dimensions}" if model and dimensions else ""

    def _store(self, state: ProactiveState) -> None:
        assert self.context is not None
        data = state.model_dump()
        self.context["proactive"] = data
        self.context.response.metadata["proactive"] = data


def _embed_text(candidate: dict) -> str:
    """Vector text for a candidate (v5.2 calibration: title+reason)."""
    title = str(candidate.get("title") or "")
    reason = str(candidate.get("reason") or "").strip()
    if reason:
        return f"{title}。{reason}"
    keywords = ", ".join(str(k) for k in candidate.get("keywords") or [])
    return f"{title} | {keywords}" if keywords else title


def _known_text(title: str, reason: str) -> str:
    """Vector text for a known topic: title+reason when available, else title."""
    reason = str(reason or "").strip()
    return f"{title}。{reason}" if reason else title


_EVIDENCE_DATE_RE = re.compile(r"(?:^|/)(\d{4}-\d{2}-\d{2})/")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
