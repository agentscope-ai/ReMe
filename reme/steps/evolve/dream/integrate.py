"""Dream unit integration step."""

import asyncio
import hashlib
import json
from pathlib import Path

from ...base_step import BaseStep
from .._evolve import agent_reply_result_text
from ....components import R
from ....enumeration import DreamBucketEnum
from ....schema import IntegrateOutcome
from .utils import llm_available, pack_paths, parse_structured_reply, state_from_context, store_state, workspace_dir

_TOOLS = ("node_search", "read", "frontmatter_read", "write", "edit", "frontmatter_update")


def _snapshot_digest(workspace: Path, digest_dir: str) -> dict[str, str]:
    """Capture digest content for side-effect recovery and notification semantics."""
    snapshot: dict[str, str] = {}
    for bucket in DreamBucketEnum:
        root = workspace / digest_dir / bucket.value
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                with path.open("rb") as file_obj:
                    digest = hashlib.file_digest(file_obj, "sha256").hexdigest()
            except OSError:
                continue
            snapshot[path.relative_to(workspace).as_posix()] = digest
    return snapshot


def _changed_digest_paths(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Return files created or changed during one integration attempt."""
    return sorted(path for path, metadata in after.items() if before.get(path) != metadata)


def _record_modified_paths(state, paths: list[str]) -> None:
    """Record content changes once while preserving discovery order."""
    for path in paths:
        if path not in state.modified_paths:
            state.modified_paths.append(path)


@R.register("dream_integrate_step")
class DreamIntegrateStep(BaseStep):
    """Integrate each extracted unit into digest memory."""

    async def execute(self):
        assert self.context is not None
        state = state_from_context(self)
        if not state.units:
            self.logger.info(f"[{self.name}] skip no units")
            return self._finish(state, True, "No dream units to integrate")
        if not llm_available(self):
            err = "no llm configured; dream integrate requires an LLM"
            state.errors.append(err)
            state.failed_units = state.units
            state.failed_paths = sorted({p for u in state.units for p in u.get("paths", [])})
            self.logger.warning(f"[{self.name}] skip no llm units={len(state.units)}")
            return self._finish(state, False, err)

        async with self._integration_lock():
            return await self._execute_locked(state)

    async def _execute_locked(self, state):
        """Integrate units while serializing digest writes for this Application."""

        workspace = Path(state.workspace).resolve() if state.workspace else workspace_dir(self)
        digest_dir = self.config_value("digest_dir")
        self.logger.info(f"[{self.name}] start units={len(state.units)} workspace={workspace} digest_dir={digest_dir}")
        for bucket in DreamBucketEnum:
            (workspace / digest_dir / bucket.value).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[{self.name}] digest dirs ready buckets={len(list(DreamBucketEnum))}")
        for i, unit in enumerate(state.units, start=1):
            await self._integrate_one(state, unit, i, workspace, digest_dir)
        state.failed_paths = sorted(set(state.failed_paths))
        answer = (
            f"Integrated {len(state.integrate_results)} unit(s); skipped {len(state.skipped_units)} unit(s); "
            f"failed {len(state.failed_units)} unit(s)"
        )
        return self._finish(state, not state.failed_units, answer)

    def _integration_lock(self) -> asyncio.Lock:
        """Return the Application-wide lock used by AutoDream integration."""
        if self.app_context is None:
            raise RuntimeError("dream_integrate_step requires app_context")
        key = "dream_integration_lock"
        lock = self.app_context.metadata.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.app_context.metadata[key] = lock
        if not isinstance(lock, asyncio.Lock):
            raise TypeError(f"app_context.metadata[{key!r}] must be an asyncio.Lock")
        return lock

    async def _integrate_one(self, state, unit: dict, index: int, workspace: Path, digest_dir: str) -> None:
        try:
            bucket = DreamBucketEnum(str(unit.get("bucket") or "")).value
        except ValueError:
            bucket = DreamBucketEnum.WIKI.value
        paths = [str(p) for p in unit.get("paths", [])]
        self.logger.info(
            f"[{self.name}] unit {index}/{len(state.units)} start "
            f"name={unit.get('name', '')!r} bucket={bucket} paths={len(paths)}",
        )
        # Keep the original baseline across the retry so a file created by attempt one
        # cannot be misclassified as a pre-existing cross-bucket UPDATE on attempt two.
        unit_before = _snapshot_digest(workspace, digest_dir)
        for attempt in range(2):
            before = _snapshot_digest(workspace, digest_dir)
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
            except Exception as e:  # noqa: BLE001
                changed = _changed_digest_paths(before, _snapshot_digest(workspace, digest_dir))
                _record_modified_paths(state, changed)
                created = len(changed) == 1 and changed[0] not in unit_before
                if len(changed) == 1 and self._valid_target(
                    workspace,
                    digest_dir,
                    bucket,
                    changed[0],
                    action="CREATE" if created else "UPDATED",
                    existed_before=changed[0] in unit_before,
                ):
                    self._record_recovered(state, unit, bucket, paths, changed[0], created, e)
                    self.logger.warning(
                        f"[{self.name}] unit {index}/{len(state.units)} recovered from file change "
                        f"after agent error target_path={changed[0]} error={type(e).__name__}: {e}",
                    )
                    return
                if attempt == 0:
                    self.logger.warning(
                        f"[{self.name}] unit {index}/{len(state.units)} attempt 1 had no recoverable file change; "
                        f"retrying once: {type(e).__name__}: {e}",
                    )
                    continue
                self._record_failure(state, unit, paths, e)
                self.logger.error(f"[{self.name}] unit {index}/{len(state.units)} failed: {type(e).__name__}: {e}")
                return

            try:
                raw_result = agent_reply_result_text(result)
                outcome = IntegrateOutcome.model_validate(parse_structured_reply(raw_result))
                if not self._valid_target(
                    workspace,
                    digest_dir,
                    bucket,
                    outcome.target_path,
                    action=outcome.action,
                    existed_before=outcome.target_path in unit_before,
                ):
                    raise ValueError(f"invalid or missing digest target_path: {outcome.target_path!r}")
            except Exception as e:  # noqa: BLE001
                changed = _changed_digest_paths(before, _snapshot_digest(workspace, digest_dir))
                _record_modified_paths(state, changed)
                created = len(changed) == 1 and changed[0] not in unit_before
                if len(changed) == 1 and self._valid_target(
                    workspace,
                    digest_dir,
                    bucket,
                    changed[0],
                    action="CREATE" if created else "UPDATED",
                    existed_before=changed[0] in unit_before,
                ):
                    self._record_recovered(state, unit, bucket, paths, changed[0], created, e)
                    self.logger.warning(
                        f"[{self.name}] unit {index}/{len(state.units)} accepted file change with invalid receipt "
                        f"target_path={changed[0]} error={type(e).__name__}: {e}",
                    )
                    return
                if attempt == 0:
                    self.logger.warning(
                        f"[{self.name}] unit {index}/{len(state.units)} attempt 1 returned an invalid receipt; "
                        "retrying once",
                    )
                    continue
                # After the bounded retry, checkpoint the source so malformed input cannot loop forever.
                self._record_skipped(state, unit, bucket, paths, e)
                self.logger.warning(
                    f"[{self.name}] unit {index}/{len(state.units)} skipped invalid receipt after retry: "
                    f"{type(e).__name__}: {e}",
                )
                return

            _record_modified_paths(
                state,
                _changed_digest_paths(unit_before, _snapshot_digest(workspace, digest_dir)),
            )
            self._append_result(
                state,
                unit,
                bucket,
                paths,
                action=outcome.action,
                target_path=outcome.target_path,
                note=outcome.note,
            )
            self.logger.info(
                f"[{self.name}] unit {index}/{len(state.units)} done "
                f"action={outcome.action} target_path={outcome.target_path}",
            )
            return

    def _finish(self, state, success: bool, answer: str):
        assert self.context is not None
        state.summary = answer
        store_state(self, state)
        self.context.response.success = success
        self.context.response.answer = answer
        self.logger.info(
            f"[{self.name}] finish success={success} integrated={len(state.integrate_results)} "
            f"failed={len(state.failed_units)}",
        )
        return self.context.response

    @staticmethod
    def _valid_target(
        workspace: Path,
        digest_dir: str,
        bucket: str,
        target_path: str,
        *,
        action: str,
        existed_before: bool,
    ) -> bool:
        target = str(target_path or "").strip().replace("\\", "/")
        parts = Path(target).parts
        if not parts or Path(target).is_absolute() or ".." in parts:
            return False
        target_bucket = next(
            (
                candidate.value
                for candidate in DreamBucketEnum
                if target.startswith(f"{digest_dir.strip('/')}/{candidate.value}/")
            ),
            None,
        )
        if target_bucket is None or not (workspace / target).is_file():
            return False
        if action == "CREATE":
            return target_bucket == bucket and not existed_before
        return existed_before

    @staticmethod
    def _unit_key(unit: dict, bucket: str, paths: list[str]) -> tuple[str, str, tuple[str, ...]]:
        return (str(unit.get("name") or unit.get("unit") or "").strip(), bucket, tuple(dict.fromkeys(paths)))

    @classmethod
    def _append_result(
        cls,
        state,
        unit: dict,
        bucket: str,
        paths: list[str],
        *,
        action: str,
        target_path: str,
        note: str,
    ) -> None:
        key = cls._unit_key(unit, bucket, paths)
        exists = any(
            cls._unit_key(result, str(result.get("bucket") or ""), [str(p) for p in result.get("paths") or []]) == key
            for result in state.integrate_results
        )
        if not exists:
            state.integrate_results.append(
                {
                    "unit": unit.get("name", ""),
                    "bucket": bucket,
                    "paths": paths,
                    "action": action,
                    "target_path": target_path,
                    "note": note,
                },
            )
        targets = state.nodes_created if action == "CREATE" else state.nodes_updated
        if target_path not in targets:
            targets.append(target_path)

    @classmethod
    def _record_recovered(
        cls,
        state,
        unit: dict,
        bucket: str,
        paths: list[str],
        target_path: str,
        created: bool,
        error: Exception,
    ) -> None:
        message = f"unit {unit.get('name', '')!r} used file changes because its agent receipt was invalid"
        if message not in state.warnings:
            state.warnings.append(message)
        cls._append_result(
            state,
            unit,
            bucket,
            paths,
            action="CREATE" if created else "UPDATED",
            target_path=target_path,
            note=f"Recovered from {type(error).__name__}: agent receipt unavailable",
        )

    @classmethod
    def _record_skipped(cls, state, unit: dict, bucket: str, paths: list[str], error: Exception) -> None:
        message = f"unit {unit.get('name', '')!r} skipped: unusable agent receipt ({type(error).__name__})"
        key = cls._unit_key(unit, bucket, paths)
        if not any(
            cls._unit_key(item, str(item.get("bucket") or ""), [str(p) for p in item.get("paths") or []]) == key
            for item in state.skipped_units
        ):
            state.skipped_units.append({**unit, "bucket": bucket, "paths": paths, "reason": message})
        if message not in state.warnings:
            state.warnings.append(message)

    @staticmethod
    def _record_failure(state, unit: dict, paths: list[str], error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        state.failed_units.append({**unit, "error": message})
        state.failed_paths.extend(path for path in paths if path not in state.failed_paths)
        state.errors.append(f"unit {unit.get('name', '')!r} failed: {message}")
