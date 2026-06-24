"""Dream unit integration step."""

import json
from pathlib import Path

from ...base_step import BaseStep
from ....components import R
from ....enumeration import DreamBucketEnum
from ....schema import IntegrateOutcome
from .utils import llm_available, pack_paths, parse_structured_reply, state_from_context, store_state, workspace_dir

_TOOLS = ("node_search", "read", "frontmatter_read", "write", "edit", "frontmatter_update")


@R.register("dream_integrate_step")
class DreamIntegrateStep(BaseStep):
    """Integrate each extracted unit into digest memory."""

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        if not state.units:
            return self._finish(state, True, "No dream units to integrate")
        if not llm_available(self):
            err = "no llm configured; dream integrate requires an LLM"
            state.errors.append(err)
            state.failed_units = state.units
            state.failed_paths = sorted({p for u in state.units for p in u.get("paths", [])})
            return self._finish(state, False, err)

        workspace = Path(state.workspace).resolve() if state.workspace else workspace_dir(self)
        digest_dir = self.config_value("digest_dir")
        for bucket in DreamBucketEnum:
            (workspace / digest_dir / bucket.value).mkdir(parents=True, exist_ok=True)
        for i, unit in enumerate(state.units, start=1):
            await self._integrate_one(state, unit, i, workspace, digest_dir)
        state.failed_paths = sorted(set(state.failed_paths))
        answer = f"Integrated {len(state.integrate_results)} unit(s); failed {len(state.failed_units)} unit(s)"
        return self._finish(state, not state.failed_units, answer)

    async def _integrate_one(self, state, unit: dict, index: int, workspace: Path, digest_dir: str) -> None:
        try:
            bucket = DreamBucketEnum(str(unit.get("bucket") or "")).value
        except ValueError:
            bucket = DreamBucketEnum.WIKI.value
        paths = [str(p) for p in unit.get("paths", [])]
        try:
            result = await self.agent_wrapper.reply(
                self.prompt_format(
                    "integrate_user_message",
                    hint=state.hint or "(none)",
                    unit_name=unit.get("name", ""),
                    unit_bucket=bucket,
                    unit_summary=unit.get("summary", ""),
                    unit_paths_json=json.dumps(paths, ensure_ascii=False, indent=2),
                    material_blob=pack_paths(workspace, paths),
                ),
                system_prompt=self.prompt_format(
                    f"integrate_system_prompt_{bucket}",
                    workspace_dir=str(workspace),
                    digest_dir=digest_dir,
                    bucket=bucket,
                ),
                job_tools=list(_TOOLS),
            )
            outcome = IntegrateOutcome.model_validate(parse_structured_reply(str(result.get("result") or "")))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            self.logger.error(f"[{self.name}] unit {index}/{len(state.units)} failed: {error}")
            state.failed_units.append({**unit, "error": error})
            state.failed_paths.extend(path for path in paths if path not in state.failed_paths)
            return

        state.integrate_results.append(
            {
                "unit": unit.get("name", ""),
                "bucket": bucket,
                "paths": paths,
                "action": outcome.action,
                "target_path": outcome.target_path,
                "note": outcome.note,
            },
        )
        (state.nodes_created if outcome.action == "CREATE" else state.nodes_updated).append(outcome.target_path)

    def _finish(self, state, success: bool, answer: str):
        assert self.context is not None
        state.summary = answer
        store_state(self, state)
        self.context.response.success = success
        self.context.response.answer = answer
        return self.context.response
