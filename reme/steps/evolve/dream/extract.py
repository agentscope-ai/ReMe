"""Global dream extract step."""

import json

from ...base_step import BaseStep
from ...file_io import refresh_day_index
from .._evolve import agent_reply_result_text
from ....components import R
from ....enumeration import DreamBucketEnum
from ....schema import DreamExtractOutput, DreamState
from .utils import (
    clean_paths,
    daily_dir,
    llm_available,
    pack_paths,
    parse_structured_reply,
    recent_dates,
    scan_day_files,
    store_state,
    today,
    workspace_dir,
)

_TOOLS = ("read",)


@R.register("dream_extract_step")
class DreamExtractStep(BaseStep):
    """Scan changed daily files and globally extract merged memory units."""

    def __init__(self, scan_days: int = 2, max_units: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.scan_days = scan_days
        self.max_units = max_units

    async def execute(self):
        assert self.context is not None
        day = today(self, str(self.context.get("date", "") or ""))
        raw_scan_days = self.context.get("scan_days", self.scan_days)
        scan_days = max(int(raw_scan_days or self.scan_days), 1)
        raw_max_units = self.context.get("max_units", self.max_units)
        max_units = max(int(raw_max_units or self.max_units), 0)
        dates = recent_dates(day, scan_days)
        hint = str(self.context.get("hint", "") or "").strip()
        daily, workspace = daily_dir(self), workspace_dir(self)
        if self.file_catalog is None:
            raise RuntimeError("dream_extract_step requires file_catalog")
        self.logger.info(
            f"[{self.name}] start date={day} dates={','.join(dates)} scan_days={scan_days} "
            f"max_units={max_units} hint={bool(hint)}",
        )
        for scan_day in dates:
            self.logger.info(f"[{self.name}] refresh index start date={scan_day} daily_dir={daily}")
            await refresh_day_index(self.file_store, scan_day, daily)
            self.logger.info(f"[{self.name}] refresh index done date={scan_day}")

        existing = self._existing(
            workspace,
            [path for scan_day in dates for path in scan_day_files(workspace, scan_day, daily)],
        )
        day_mds = {f"{daily}/{scan_day}.md" for scan_day in dates}
        day_prefixes = tuple(f"{daily}/{scan_day}/" for scan_day in dates)
        nodes = await self.file_catalog.get_nodes()
        # Older Auto Dream versions checkpointed generated interests files.
        # Remove every such watermark from the dream catalog, not only entries
        # inside the current scan window. The exposure files themselves remain
        # untouched and are owned by the proactive refresh pipeline.
        legacy_interests = sorted(
            {n.path for n in nodes if n.path.startswith(f"{daily}/") and n.path.endswith("/interests.yaml")},
        )
        indexed_all = {
            n.path: n.st_mtime
            for n in nodes
            if n.path not in legacy_interests and (n.path in day_mds or n.path.startswith(day_prefixes))
        }
        indexed = {path: mt for path, mt in indexed_all.items() if path in existing}
        changed = [rel for rel, mt in existing.items() if indexed.get(rel) != mt]
        unchanged = [rel for rel, mt in existing.items() if indexed.get(rel) == mt]
        deleted = sorted((indexed_all.keys() - set(existing)) | set(legacy_interests))
        self.logger.info(
            f"[{self.name}] scan summary existing={len(existing)} indexed={len(indexed)} "
            f"changed={len(changed)} unchanged={len(unchanged)} deleted={len(deleted)}",
        )
        if deleted:
            self.logger.info(f"[{self.name}] catalog delete start paths={len(deleted)}")
            await self.file_catalog.delete(deleted)
            self.logger.info(f"[{self.name}] catalog delete done paths={len(deleted)}")

        state = DreamState(
            date=day,
            dates=dates,
            scan_days=scan_days,
            hint=hint,
            daily_dir=daily,
            workspace=str(workspace),
            files_scanned=len(existing),
            files_unchanged=len(unchanged),
            files_changed=len(changed),
            files_deleted=len(deleted),
            changed_paths=changed,
            unchanged_paths=unchanged,
            deleted_paths=deleted,
            existing=existing,
            indexed=indexed,
        )
        if not changed:
            self.logger.info(f"[{self.name}] skip no changed input dates={','.join(dates)}")
            return self._finish(state, True, f"No changed dream input for {', '.join(dates)}")
        if not llm_available(self):
            state.errors.append("no llm configured; dream extract requires an LLM")
            state.failed_paths = list(changed)
            self.logger.warning(f"[{self.name}] skip no llm changed={len(changed)}")
            return self._finish(state, False, state.errors[-1])

        self.logger.info(f"[{self.name}] agent start changed={len(changed)} dates={len(dates)}")
        user_message = self.prompt_format(
            "extract_user_message",
            date=day,
            dates_json=json.dumps(dates, ensure_ascii=False, indent=2),
            hint=hint or "(none)",
            max_units=max_units,
            changed_paths_json=json.dumps(changed, ensure_ascii=False, indent=2),
            material_blob=pack_paths(workspace, changed),
        )
        retry_feedback = ""
        output_schema = DreamExtractOutput.model_json_schema()
        output_schema["required"] = ["units"]
        for attempt in range(2):
            try:
                result = await self.agent_wrapper.reply(
                    user_message + retry_feedback,
                    system_prompt=self.prompt_format(
                        "extract_system_prompt",
                        workspace_dir=str(workspace),
                        buckets=", ".join(bucket.value for bucket in DreamBucketEnum),
                        max_units=max_units,
                        output_schema_json=json.dumps(output_schema, ensure_ascii=False, indent=2),
                    ),
                    job_tools=list(_TOOLS),
                )
                self.logger.info(f"[{self.name}] agent done has_result={bool(result.get('result'))}")
                raw_result = agent_reply_result_text(result)
                state.extract_summary = raw_result
                meta = parse_structured_reply(raw_result, strict=True)
                self.clean_output(state, meta, max_units=max_units)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 0:
                    retry_feedback = "\n\n" + self.prompt_format(
                        "extract_retry_message",
                        error=f"{type(e).__name__}: {e}",
                    )
                    self.logger.warning(
                        f"[{self.name}] extract attempt 1 returned no usable receipt; retrying once: "
                        f"{type(e).__name__}: {e}",
                    )
                    continue
                error = f"dream extract agent failed after retry: {type(e).__name__}: {e}"
                state.errors.append(error)
                state.failed_paths = list(changed)
                self.logger.error(f"[{self.name}] {error}")
                return self._finish(state, False, error)

        self.logger.info(f"[{self.name}] parse done keys={','.join(sorted(meta.keys())) if meta else '(none)'}")
        answer = f"Extracted {len(state.units)} unit(s) from {len(changed)} changed file(s) across {len(dates)} day(s)"
        return self._finish(state, True, answer)

    def _existing(self, workspace, files: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for rel in files:
            try:
                out[rel] = (workspace / rel).stat().st_mtime
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {rel}: {e}")
        return out

    def clean_output(self, state: DreamState, meta: dict, max_units: int | None = None) -> None:
        """Validate the whole receipt before accepting any units or applying the cap."""
        units = meta.get("units") if "units" in meta else meta.get("memory_units")
        if not isinstance(units, list):
            raise ValueError("expected a units list (use units: [] only when nothing is worth remembering)")
        normalized = []
        for raw in units:
            if isinstance(raw, dict) and isinstance(raw.get("bucket"), str):
                raw = {**raw, "bucket": raw["bucket"].strip()}
            if (
                isinstance(raw, dict)
                and "bucket" in raw
                and raw["bucket"] not in {bucket.value for bucket in DreamBucketEnum}
            ):
                self.logger.warning(
                    f"[{self.name}] unit {raw.get('name')!r} emitted an unknown bucket; routing to wiki",
                )
                raw = {**raw, "bucket": DreamBucketEnum.WIKI.value}
            normalized.append(raw)
        output = DreamExtractOutput.model_validate({"units": normalized})
        allowed = set(state.changed_paths)
        cleaned = []
        for index, unit in enumerate(output.units):
            name, summary = unit.name.strip(), unit.summary.strip()
            paths = clean_paths(unit.paths, allowed)
            if not name or not summary or not paths:
                raise ValueError(f"units[{index}] requires non-empty name, summary, and paths from changed_paths")
            cleaned.append({"name": name, "bucket": unit.bucket.value, "summary": summary, "paths": paths})
        state.units.extend(cleaned if max_units is None else cleaned[: max(max_units - len(state.units), 0)])

    def _finish(self, state: DreamState, success: bool, answer: str):
        assert self.context is not None
        state.summary = answer
        store_state(self, state)
        self.context.response.success = success
        self.context.response.answer = answer
        self.logger.info(f"[{self.name}] finish success={success} answer={answer!r}")
        return self.context.response
