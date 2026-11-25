"""Own one application process and commit settings after it reports readiness."""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import psutil
from cryptography.fernet import Fernet

from .config import Profile, SECRET_FIELDS, Settings, SettingsRequest
from .storage import OperationHistory, SettingsStore, write_private


class OperationConflict(RuntimeError):
    """Another operation or a newer configuration prevents the requested edit."""


@contextmanager
def lock_directory(directory: Path):
    """Keep a deployment's persisted state under one live owner."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name != "nt":
        directory.chmod(0o700)
    with (directory / "lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt  # pylint: disable=import-outside-toplevel

                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # pylint: disable=import-outside-toplevel

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"ReMe is already using {directory}") from error
        yield


class Worker:
    """Hold the only launch pipe and private readiness file for one child."""

    def __init__(self):
        self.directory = TemporaryDirectory(prefix="reme-worker-")  # pylint: disable=consider-using-with
        self.process: asyncio.subprocess.Process | None = None
        self.port: int | None = None

    async def launch(self, request: dict, environment: dict[str, str]) -> dict:
        """Start the child and wait for its readiness report."""
        result = Path(self.directory.name) / "result.json"
        system = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        )
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "reme.daemon.worker", stdin=asyncio.subprocess.PIPE, env=environment, **system
        )
        self.process.stdin.write((json.dumps({**request, "result": str(result)}) + "\n").encode())
        await self.process.stdin.drain()
        async with asyncio.timeout(120):
            while not result.is_file():
                if self.process.returncode is not None:
                    raise RuntimeError("ReMe exited before reporting readiness. See the service logs.")
                await asyncio.sleep(0.05)
        payload = json.loads(result.read_text(encoding="utf-8"))
        if "error" in payload:
            raise RuntimeError(payload["error"])
        self.port = payload.get("port")
        return payload

    async def stop(self) -> None:
        """Close the parent pipe, reap the child and clean up the readiness file."""
        try:
            if self.process is None:
                return
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), 30)
            except asyncio.TimeoutError:
                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                await self.process.wait()
        finally:
            self.directory.cleanup()


class Supervisor:
    """Serialize complete settings changes, first setup and process recovery."""

    def __init__(self, directory: Path, arguments: dict, environment: dict[str, str]):
        self.directory = directory
        self.arguments = arguments
        self.environment = {
            key: value for key, value in environment.items() if key not in {"REME_ENCRYPTION_KEY", "REME_ADMIN_TOKEN"}
        }
        self.store = SettingsStore(directory)
        self.state_file = directory / "state.json"
        self.error: str | None = None
        self.load_error: str | None = None
        self.state_error: str | None = None
        try:
            self.active = self.store.load()
        except ValueError as error:
            self.active = None
            self.load_error = self.error = str(error)
        self.operations: dict[str, dict] = {}
        try:
            saved = OperationHistory.model_validate_json(self.state_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except (ValueError, OSError):
            self.state_error = "Cannot read the ReMe operation history"
        else:
            self.operations = saved.model_dump()["operations"]
        self.worker: Worker | None = None
        self.view: dict = {}
        self.task: asyncio.Task | None = None
        self.admission = asyncio.Lock()
        for operation in self.operations.values():
            if operation["phase"] in {"accepted", "restarting"}:
                operation.update(phase="failed", error="Supervisor stopped before the operation completed")

    def _persist(self) -> None:
        retained = dict(list(self.operations.items())[-32:])
        write_private(self.state_file, json.dumps({"operations": retained}, ensure_ascii=False) + "\n")
        self.operations = retained
        self.state_error = None

    @property
    def port(self) -> int | None:
        """Return the listening port of the current live worker."""
        if self.worker and self.worker.process.returncode is None:
            return self.worker.port
        return None

    def settings(self) -> dict:
        """Return ordinary values, capabilities and credential presence."""
        profile = self.active or Profile()
        return {
            "revision": profile.revision,
            **profile.settings.public_values(),
            **self.view,
            "initialized": self.store.path.is_file(),
            "settings_error": self.load_error,
        }

    def status(self, operation_id: str | None) -> dict:
        """Return setup needs independently of business service availability."""
        return {
            "ready": self.port is not None,
            "needs_setup": bool(self.view.get("missing_fields")) and not self.load_error,
            "missing_fields": self.view.get("missing_fields", []),
            "revision": self.active.revision if self.active else 0,
            "error": self.error or self.state_error,
            "operation": (
                self.operations.get(operation_id) if operation_id else next(reversed(self.operations.values()), None)
            ),
        }

    async def _launch(self, settings: Settings | None, mode: str) -> tuple[Worker, dict]:
        worker = Worker()
        report_key = Fernet.generate_key()
        try:
            payload = await worker.launch(
                {
                    "mode": mode,
                    "arguments": self.arguments,
                    "settings": settings.private_values() if settings else None,
                    "directory": str(self.directory),
                    "report_key": report_key.decode("ascii"),
                },
                self.environment,
            )
            if "settings" in payload:
                payload["settings"] = json.loads(Fernet(report_key).decrypt(payload["settings"].encode("ascii")))
        except BaseException:
            await worker.stop()
            raise
        if mode != "serve":
            await worker.stop()
        return worker, payload

    async def _start(self, profile: Profile) -> None:
        self.worker, payload = await self._launch(profile.settings, "serve")
        self.view = payload["view"]
        self.error = None

    async def _stop(self) -> None:
        worker, self.worker = self.worker, None
        if worker:
            await worker.stop()

    async def _boot(self) -> None:
        try:
            await self._stop()
            if self.active is None:
                _, payload = await self._launch(None, "import")
                self.active = Profile(settings=Settings.model_validate(payload["settings"]))
            else:
                _, payload = await self._launch(self.active.settings, "check")
            self.view = payload["view"]
            if self.view["missing_fields"]:
                return
            prepared = None if self.store.path.is_file() else self.store.prepare(self.active)
            await self._start(self.active)
            if prepared is not None:
                self.store.commit(prepared)
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            await self._stop()
            self.error = str(error) or "ReMe startup timed out"

    async def watch(self) -> None:
        """Recover process failures while waiting quietly for missing configuration."""
        failures = 0
        while True:
            startup = None
            async with self.admission:
                if (
                    not self.load_error
                    and not self.view.get("missing_fields")
                    and self.port is None
                    and (self.task is None or self.task.done())
                ):
                    startup = self.task = asyncio.create_task(self._boot())
            if startup is not None:
                await asyncio.shield(startup)
                failures = 0 if self.port is not None else failures + 1
            await asyncio.sleep(min(30, 2 ** min(failures, 5)))

    def _candidate(self, request: SettingsRequest | None) -> Profile:
        if self.load_error:
            raise RuntimeError(self.load_error)
        profile = self.active or Profile()
        if request is None:
            return profile
        if request.revision != profile.revision:
            raise OperationConflict("The ReMe settings have changed. Reload them before saving.")
        changes = request.private_values()
        changes = {
            key: value for key, value in changes.items() if key in request.model_fields_set and key != "revision"
        }
        for key in SECRET_FIELDS:
            if not changes.get(key):
                changes.pop(key, None)
        values = {**profile.settings.private_values(), **changes}
        return Profile(revision=profile.revision + 1, settings=Settings.model_validate(values))

    async def test_model(self, request: SettingsRequest) -> None:
        """Test proposed model settings without saving or restarting the active process."""
        async with self.admission:
            profile = self._candidate(request)
        await self._launch(profile.settings, "test")

    async def request(self, request: SettingsRequest | None) -> dict:
        """Validate a complete candidate before replacing the business worker."""
        async with self.admission:
            if self.task and not self.task.done():
                raise OperationConflict("ReMe is already restarting")
            profile = self._candidate(request)
            if profile.settings.missing_fields():
                raise RuntimeError("Missing required settings: " + ", ".join(profile.settings.missing_fields()))
            _, payload = await self._launch(profile.settings, "check")
            if payload["view"]["missing_fields"]:
                raise RuntimeError("Missing required settings: " + ", ".join(payload["view"]["missing_fields"]))
            prepared = self.store.prepare(profile)
            identity = uuid4().hex
            operation = {"id": identity, "phase": "accepted", "error": None}
            self.operations[identity] = operation
            try:
                self._persist()
            except OSError as error:
                self.operations.pop(identity)
                self.state_error = "Cannot save the ReMe operation history"
                raise RuntimeError(self.state_error) from error
            self.task = asyncio.create_task(self._replace(profile, prepared, operation))
            return dict(operation)

    async def _replace(self, candidate: Profile, prepared: str, operation: dict) -> None:
        operation.update(phase="restarting")
        previous = self.active
        try:
            await self._stop()
            await self._start(candidate)
            self.store.commit(prepared)
            self.active = candidate
            operation.update(phase="succeeded")
        except (OSError, RuntimeError, TimeoutError) as error:
            self.active = previous
            failure = str(error) or "ReMe startup timed out"
            await self._stop()
            if previous:
                self.view.update(previous.settings.public_values(), missing_fields=previous.settings.missing_fields())
            if previous and not previous.settings.missing_fields():
                try:
                    await self._start(previous)
                except (OSError, RuntimeError, TimeoutError):
                    pass
            self.error = failure
            operation.update(phase="failed", error=failure)
        try:
            self._persist()
        except OSError:
            self.state_error = "Cannot save the ReMe operation history"

    async def close(self) -> None:
        """Cancel pending work and stop the owned worker."""
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self._stop()
