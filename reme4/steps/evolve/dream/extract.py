"""Global dream extract step."""

import json

from ...base_step import BaseStep
from ...file_io import refresh_day_index
from ....components import R
from .schema import BUCKETS, DreamState
from .utils import (
    clean_paths,
    daily_dir,
    llm_available,
    pack_paths,
    parse_structured_reply,
    scan_day_files,
    store_state,
    today,
    vault_dir,
)

_TOOLS = ("read",)


@R.register("dream_extract_step")
class DreamExtractStep(BaseStep):
    """Scan changed daily files and globally extract merged units/topics."""

    def __init__(self, topic_session_id: str = "interests", **kwargs):
        super().__init__(**kwargs)
        self.topic_session_id = topic_session_id

    async def execute(self):
        assert self.context is not None
        day = today(self, str(self.context.get("date", "") or ""))
        hint = str(self.context.get("hint", "") or "").strip()
        daily, vault = daily_dir(self), vault_dir(self)
        if self.file_catalog is None:
            raise RuntimeError("dream_extract_step requires file_catalog")
        await refresh_day_index(self.file_store, day, daily)

        existing = self._existing(vault, scan_day_files(vault, day, daily, f"{self.topic_session_id}.yaml"))
        interests_rel = f"{daily}/{day}/{self.topic_session_id}.yaml"
        day_md, day_prefix = f"{daily}/{day}.md", f"{daily}/{day}/"
        nodes = await self.file_catalog.get_nodes()
        indexed_all = {n.path: n.st_mtime for n in nodes if n.path == day_md or n.path.startswith(day_prefix)}
        indexed = {path: mt for path, mt in indexed_all.items() if path != interests_rel}
        changed = [rel for rel, mt in existing.items() if indexed.get(rel) != mt]
        unchanged = [rel for rel, mt in existing.items() if indexed.get(rel) == mt]
        protected = set(existing) | ({interests_rel} if (vault / interests_rel).is_file() else set())
        deleted = sorted(indexed_all.keys() - protected)
        if deleted:
            await self.file_catalog.delete(deleted)

        state = DreamState(
            date=day,
            hint=hint,
            daily_dir=daily,
            vault=str(vault),
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
            return self._finish(state, True, f"No changed dream input for {day}")
        if not llm_available(self):
            state.errors.append("no llm configured; dream extract requires an LLM")
            return self._finish(state, False, state.errors[-1])

        result = await self.agent_wrapper.reply(
            self.prompt_format(
                "extract_user_message",
                date=day,
                hint=hint or "(none)",
                changed_paths_json=json.dumps(changed, ensure_ascii=False, indent=2),
                material_blob=pack_paths(vault, changed),
            ),
            system_prompt=self.prompt_format("extract_system_prompt", vault_dir=str(vault), buckets=", ".join(BUCKETS)),
            job_tools=list(_TOOLS),
        )
        meta = parse_structured_reply(str(result.get("result") or ""))
        self._clean_output(state, meta)
        state.extract_summary = str(result.get("result") or "").strip()
        answer = f"Extracted {len(state.units)} unit(s), {len(state.topics)} topic(s)"
        answer = f"{answer} from {len(changed)} changed file(s)"
        return self._finish(state, True, answer)

    def _existing(self, vault, files: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for rel in files:
            try:
                out[rel] = (vault / rel).stat().st_mtime
            except OSError as e:
                self.logger.error(f"[{self.name}] stat failed on {rel}: {e}")
        return out

    def _clean_output(self, state: DreamState, meta: dict) -> None:
        allowed = set(state.changed_paths)
        for raw in meta.get("units") or meta.get("memory_units") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip()
            summary = str(raw.get("summary") or "").strip()
            bucket = str(raw.get("bucket") or "").strip()
            paths = clean_paths(raw.get("paths"), allowed)
            if not name or not summary or not paths:
                continue
            if bucket not in BUCKETS:
                self.logger.warning(f"[{self.name}] unit {name!r} emitted bucket {bucket!r}; routing to wiki")
                bucket = "wiki"
            state.units.append({"name": name, "bucket": bucket, "summary": summary, "paths": paths})
        for raw in meta.get("topics") or []:
            topic = self._clean_topic(raw, allowed)
            if topic:
                state.topics.append(topic)

    @staticmethod
    def _clean_topic(raw, allowed: set[str]) -> dict:
        if not isinstance(raw, dict):
            return {}
        title = str(raw.get("title") or "").strip()
        reason = str(raw.get("reason") or "").strip()
        paths = clean_paths(raw.get("paths"), allowed)
        if not title or not reason or not paths:
            return {}
        keywords = raw.get("keywords") or []
        cleaned_keywords = [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []
        return {
            "title": title,
            "reason": reason,
            "evidence": str(raw.get("evidence") or "").strip(),
            "keywords": cleaned_keywords,
            "paths": paths,
        }

    def _finish(self, state: DreamState, success: bool, answer: str):
        assert self.context is not None
        state.summary = answer
        store_state(self, state)
        self.context.response.success = success
        self.context.response.answer = answer
        return self.context.response
