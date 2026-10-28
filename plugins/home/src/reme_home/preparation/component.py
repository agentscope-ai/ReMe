"""One-time experiment preparation, shared by Python and the Home Job."""

import asyncio
from pathlib import Path
from threading import Event
import shutil
from datetime import datetime, timezone
from importlib.resources import files
from uuid import uuid4

import yaml

from reme.components.base_component import BaseComponent
from reme.steps.base_step import BaseStep, Ref

from .code import prepare_code
from .data import prepare_dataset
from .data.basic import derive_debug_cases
from .data.models import fingerprint
from .spec import PreparationSpec
from ..judge import HomeJudge
from .transaction import (
    TRANSACTION,
    check_targets,
    publish,
    recover_preparation,
    rollback,
    write_json,
    sync_directory,
)
from ..workspace import HomeWorkspace, _initialization_lock, _validate_directory


def _source_path(root: Path, value: str) -> Path:
    if value.startswith("~"):
        raise ValueError("Home-relative paths are unsupported; use an absolute or workspace-relative path")
    path = Path(value)
    if not path.is_absolute() and ".." in path.parts:
        raise ValueError("Relative source paths must stay inside the workspace")
    resolved = (root / path).resolve(strict=True)
    if not path.is_absolute() and not resolved.is_relative_to(root):
        raise ValueError("Relative source paths must stay inside the workspace")
    return resolved


class HomePreparation(BaseComponent):
    """Prepare once under Home's process lock; never reset an existing experiment."""

    component_type = "home.preparation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.workspace = self.bind("default", HomeWorkspace, optional=False)
        self.judge = self.bind("default", HomeJudge, optional=False)
        self._pending: set[asyncio.Task] = set()

    async def _close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def prepare(self, *, spec: dict | None = None, spec_path: str | None = None) -> dict:
        """Keep blocking data/Git work off the event loop and await rollback on cancellation."""
        stop = Event()
        task = asyncio.create_task(asyncio.to_thread(self._prepare, spec=spec, spec_path=spec_path, stop=stop))
        self._pending.add(task)
        cancelled = False
        try:
            while True:
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    if task.cancelled():
                        raise
                    cancelled = True
                    stop.set()
                except InterruptedError:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
        finally:
            self._pending.discard(task)

    def _prepare(self, *, spec: dict | None, spec_path: str | None, stop: Event) -> dict:
        """Prepare a started application's workspace from exactly one input form.

        Local YAML paths must stay inside the workspace. Explicit dataset/repository
        source paths may reference external read-only inputs. No source code is run.
        """
        if not self.is_started:
            raise RuntimeError("Start the Home application before preparation")
        workspace = self.workspace_path.resolve()
        root = self.workspace.home_path
        state = self.workspace.state_path
        _validate_directory(workspace, state)
        with _initialization_lock(state / "workspace.lock"):
            recover_preparation(root, state)
            marker = state / "preparation.json"
            if marker.exists() or marker.is_symlink():
                raise RuntimeError("Home workspace is already prepared; use a new workspace")
            if (spec is None) == (spec_path is None):
                raise ValueError("Provide exactly one of spec or spec_path")
            if spec_path is not None:
                path = _source_path(workspace, spec_path)
                if not path.is_relative_to(workspace) or not path.is_file():
                    raise ValueError("spec_path must be a file inside the workspace")
                spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = PreparationSpec.model_validate(spec)
            self.judge.require_benchmark(config.benchmark.name)
            # Resolve explicitly supplied source inputs before creating transaction state.
            source = _source_path(workspace, config.benchmark.source)
            repository = _source_path(workspace, config.runtime.source_repo)
            evidence = (
                _source_path(workspace, config.benchmark.evidence_source) if config.benchmark.evidence_source else None
            )
            config.benchmark.source = str(source)
            config.runtime.source_repo = str(repository)
            config.benchmark.evidence_source = str(evidence) if evidence else None
            existed = check_targets(root, state)
            transaction = state / TRANSACTION
            transaction.mkdir()
            try:
                sync_directory(state)
                result = self._stage(config, transaction, source, repository, evidence)

                def check_cancelled():
                    if stop.is_set():
                        raise InterruptedError("Home preparation cancelled")

                check_cancelled()
                # Catch conflicts introduced while staging before replacing any target.
                if check_targets(root, state) != existed:
                    raise RuntimeError("Preparation targets changed during staging")
                publish(root, state, existed, check_cancelled)
            except BaseException:
                rollback(root, state)
                raise
            # Completion is durable. Cleanup failure must not report a failed preparation.
            try:
                shutil.rmtree(transaction)
            except OSError:
                self.logger.warning("Preparation committed; transaction cleanup will retry at next startup")
            return result

    @staticmethod
    def _stage(config, transaction, source, repository, evidence) -> dict:
        dataset = prepare_dataset(
            transaction / "output-2",
            config.benchmark.name,
            source,
            train_case_ids=config.benchmark.case_ids,
            variant=config.benchmark.variant,
            evidence_source=evidence,
        )
        # Materialize the actual all-cases selection as part of the durable experiment contract.
        config.benchmark.case_ids = dataset.case_ids
        config.benchmark.all_cases = False
        if config.debug.enabled:
            derive_debug_cases(
                transaction / "output-2",
                transaction / "output-3",
                sessions_per_case=config.debug.sessions_per_case,
                messages_per_session=config.debug.messages_per_session,
            )
        else:
            (transaction / "output-3/cases").mkdir(parents=True)
        code = prepare_code(config, repository, transaction / "output-0", transaction / "output-1")
        config.runtime.source_ref = code["source_commit"]
        domain = config.model_dump(mode="json")
        (transaction / "output-4").write_text(yaml.safe_dump(domain, sort_keys=False), encoding="utf-8")
        docs = transaction / "output-5"
        docs.mkdir()
        resources = files("reme_home").joinpath("runtime_docs")
        for name in ("architecture-guide.md", "validation-artifact-guide.md"):
            (docs / name).write_bytes(resources.joinpath(name).read_bytes())
        result = {
            "schema_version": 1,
            "preparation_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": fingerprint(domain),
            "dataset": dataset.model_dump(mode="json"),
            "code": code,
        }
        write_json(transaction / "output-6", result)
        return result


class HomePrepareStep(BaseStep):
    """Accept a workspace YAML file or a Python configuration dictionary."""

    preparation: HomePreparation = Ref(HomePreparation, "home.preparation")

    async def execute(self):
        """Prepare once and return the persisted completion information."""
        if self.context.get("_allowed_paths") is not None:
            raise ValueError("home_prepare requires unrestricted access to its workspace output directories")
        result = await self.preparation.prepare(
            spec=self.context.get("spec"),
            spec_path=self.context.get("spec_path"),
        )
        self.context.response.answer = result
        return self.context.response
