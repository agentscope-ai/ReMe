"""Application-owned Codex process, concurrent sessions, and local job tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

from .base_component import BaseComponent
from .component_registry import R
from .service.scoped_mcp import ScopedMCP
from ..constants import normalize_connect_host
from ..enumeration import ComponentEnum

if TYPE_CHECKING:
    from openai_codex import AsyncCodex, AsyncThread, CodexConfig


@dataclass
class _ThreadState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    revision: int = 0
    binding: str | None = None
    token: str | None = None
    completion: asyncio.Future[None] | None = None
    operation_started: bool = False


@R.register("sdk")
class CodexAppServer(BaseComponent):
    """Share one SDK client across wrappers with the same account and launch settings."""

    component_type = ComponentEnum.CODEX_APP_SERVER

    def __init__(
        self,
        auth_mode: str = "auto",
        api_key: str = "",
        base_url: str = "",
        codex_home: str | Path | None = None,
        codex_bin: str | None = None,
        config_overrides: list[str] | tuple[str, ...] = (),
        model_catalog: dict[str, Any] | None = None,
        cwd: str | Path | None = None,
        experimental_api: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if auth_mode not in {"auto", "api_key", "oauth"}:
            raise ValueError("auth_mode must be one of: auto, api_key, oauth")
        self._auth_mode = auth_mode
        self._api_key = api_key
        self._base_url = base_url
        self._codex_bin = codex_bin
        self._config_overrides = tuple(config_overrides)
        self._model_catalog = model_catalog
        self._cwd = cwd
        self._experimental_api = experimental_api
        self._codex_home = codex_home
        self._codex: AsyncCodex | None = None
        self._notifications: asyncio.Task | None = None
        self._client_lock = asyncio.Lock()
        self._threads: dict[str, _ThreadState] = {}
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._failed = False
        self._http_url: str | None = None
        self.jobs_mcp = ScopedMCP()

    @property
    def session_path(self) -> Path:
        """Persistent Codex state; different server configurations have separate homes."""
        if self._codex_home:
            path = Path(self._codex_home).expanduser()
            return path if path.is_absolute() else self.workspace_path / path
        directory = self.app_context.app_config.mem_session_dir if self.app_context else "mem_session"
        return self.workspace_path / directory / "codex" / self.name

    def _client_config(self) -> CodexConfig:
        from openai_codex import CodexConfig

        environment = dict(self.app_context.app_config.environment) if self.app_context else {}
        environment["CODEX_HOME"] = str(self.session_path)
        overrides = list(self._config_overrides)
        api_login = self._auth_mode != "oauth" and bool(self._api_key)
        if self._auth_mode == "api_key" and not api_login:
            raise ValueError("auth_mode='api_key' requires a non-empty API key")
        self.session_path.mkdir(parents=True, exist_ok=True)
        if self._model_catalog is not None:
            if any(option.partition("=")[0].strip() == "model_catalog_json" for option in overrides):
                raise ValueError("Configure either model_catalog or config_overrides.model_catalog_json, not both")
            catalog_path = self.session_path / "reme-model-catalog.json"
            catalog_path.write_text(json.dumps(self._model_catalog, ensure_ascii=False), encoding="utf-8")
            overrides.append(f"model_catalog_json={json.dumps(str(catalog_path))}")
        if api_login and self._base_url:
            overrides.append(f"openai_base_url={json.dumps(self._base_url)}")
        overrides.append(f"forced_login_method={json.dumps('api' if api_login else 'chatgpt')}")
        cwd = Path(self._cwd).expanduser() if self._cwd else self.workspace_path
        if not cwd.is_absolute():
            cwd = self.workspace_path / cwd
        return CodexConfig(
            codex_bin=self._codex_bin,
            config_overrides=tuple(overrides),
            cwd=str(cwd),
            env=environment,
            client_name="reme",
            client_title="ReMe",
            experimental_api=self._experimental_api,
        )

    async def _get_client(self) -> AsyncCodex:
        from openai_codex import AsyncCodex

        async with self._client_lock:
            if self._failed:
                if self._active > 1:
                    raise RuntimeError("Codex disconnected; wait for its active requests to finish before reconnecting")
                await self._close_client()
                self._threads.clear()
                self.jobs_mcp.scopes.clear()
                self._failed = False
            if self._codex is None:
                codex = AsyncCodex(self._client_config())
                try:
                    if self._auth_mode != "oauth" and self._api_key:
                        await codex.login_api_key(self._api_key)
                    elif (await codex.account()).account is None:
                        raise RuntimeError(f"No ChatGPT OAuth login found in CODEX_HOME: {self.session_path}")
                except BaseException:
                    await codex.close()
                    raise
                self._codex = codex
                self._notifications = asyncio.create_task(self._watch_notifications(codex))
            return self._codex

    async def _watch_notifications(self, codex: AsyncCodex) -> None:
        """Route SDK status events to the operation that owns each conversation."""
        try:
            while self._codex is codex:
                event = await codex._client.next_notification()  # pylint: disable=protected-access
                if event.method != "thread/status/changed":
                    continue
                state = self._threads.get(event.payload.thread_id)
                if state is None or state.completion is None or state.completion.done():
                    continue
                if event.payload.status.root.type == "active":
                    state.operation_started = True
                elif state.operation_started:
                    state.completion.set_result(None)
        except Exception as exc:
            if self._codex is codex:
                self._failed = True
            for state in self._threads.values():
                if state.completion is not None and not state.completion.done():
                    state.completion.set_exception(exc)

    async def _close_client(self) -> None:
        codex, self._codex = self._codex, None
        try:
            if codex is not None:
                await codex.close()
        finally:
            if self._notifications is not None:
                self._notifications.cancel()
                with suppress(asyncio.CancelledError):
                    await self._notifications
                self._notifications = None

    async def _mcp_url(self) -> str:
        from .service.http_service import HttpService

        service = self.app_context.service if self.app_context else None
        if (
            isinstance(service, HttpService)
            and service.service is not None
            and any(getattr(route, "endpoint", None) is self.jobs_mcp for route in service.service.routes)
        ):
            if self._http_url is None:
                host = normalize_connect_host(service.host)
                host = "::1" if host == "::" else host
                # Background jobs can run before uvicorn finishes binding the main listener.
                async with asyncio.timeout(30):
                    while True:
                        try:
                            _, writer = await asyncio.open_connection(host, service.port)
                        except OSError:
                            await asyncio.sleep(0.01)
                        else:
                            writer.close()
                            await writer.wait_closed()
                            break
                authority = f"[{host}]" if ":" in host else host
                self._http_url = f"http://{authority}:{service.port}{self.jobs_mcp.path}"
            return self._http_url
        return await self.jobs_mcp.listen()

    async def _prune_threads(self, codex: AsyncCodex) -> None:
        """Revoke idle tool bindings only after the server confirms their threads unloaded."""
        from openai_codex.generated.v2_all import ThreadLoadedListResponse

        candidates = {key: (state, state.revision) for key, state in self._threads.items() if not state.users}
        if not candidates:
            return
        try:
            loaded: set[str] = set()
            cursor = None
            while True:
                page = await codex._client.request(  # pylint: disable=protected-access
                    "thread/loaded/list",
                    {"cursor": cursor},
                    response_model=ThreadLoadedListResponse,
                )
                loaded.update(page.data)
                cursor = page.next_cursor
                if not cursor:
                    break
        except Exception as exc:
            self.logger.warning(f"Failed to list loaded Codex threads: {exc}")
            return
        for key, (state, revision) in candidates.items():
            if key not in loaded and not state.users and state.revision == revision and self._threads.get(key) is state:
                self._threads.pop(key)
                self.jobs_mcp.scopes.pop(state.token, None)

    async def wait_idle(self, thread: AsyncThread) -> None:
        """Keep the thread lock through cancellation or asynchronous compaction."""
        state = self._threads[thread.id]
        if state.completion is not None:
            await asyncio.shield(state.completion)
            state.completion = None
        async with asyncio.timeout(30):
            while (await thread.read()).thread.status.root.type == "active":
                await asyncio.sleep(0.05)

    @asynccontextmanager
    async def _operation(self, thread: AsyncThread) -> AsyncIterator[None]:
        """Register before enqueueing work; retain the barrier if the caller is cancelled."""
        state = self._threads[thread.id]
        state.operation_started = False
        state.completion = asyncio.get_running_loop().create_future()
        state.completion.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        try:
            yield
        except Exception:
            state.completion.cancel()
            state.completion = None
            raise

    async def turn(self, thread: AsyncThread, inputs: Any, **kwargs):
        """Start a normal SDK turn with completion tracking owned by this server."""
        async with self._operation(thread):
            return await thread.turn(inputs, **kwargs)

    async def compact(self, thread: AsyncThread) -> None:
        """Enqueue native compaction; the thread lease waits for its completion event."""
        async with self._operation(thread):
            await thread.compact()

    @asynccontextmanager
    async def thread(
        self,
        thread_id: str = "",
        *,
        fork: bool = False,
        options: dict[str, Any] | None = None,
        jobs: list[Any] | None = None,
        injected: dict[str, Any] | None = None,
        mcp_startup_timeout: float = 30,
        mcp_tool_timeout: float = 300,
    ) -> AsyncIterator[AsyncThread]:
        """Serialize one conversation while allowing independent conversations to run concurrently."""
        from openai_codex import TransportClosedError

        await self.start()
        self._active += 1
        self._idle.clear()
        state = None
        created_token = None
        thread = None
        codex = None
        try:
            codex = await self._get_client()
            await self._prune_threads(codex)
            state = self._threads.setdefault(thread_id, _ThreadState()) if thread_id and not fork else _ThreadState()
            state.users += 1
            state.revision += 1
            async with state.lock:
                options = dict(options or {})
                if jobs is not None:
                    injected = json.loads(json.dumps(injected or {}))
                    binding = json.dumps({"jobs": [job.name for job in jobs], "injected": injected}, sort_keys=True)
                    if state.binding is not None and state.binding != binding:
                        raise ValueError(
                            "job_tools, tool_context_id and injected_job_kwargs cannot change on a loaded thread",
                        )
                    state.binding = binding
                    config = dict(options.get("config") or {})
                    servers = dict(config.get("mcp_servers") or {})
                    if "reme_jobs" in servers:
                        raise ValueError("mcp_servers.reme_jobs is reserved for ReMe job tools")
                    # Codex validates the transport even when a server is disabled.
                    servers["reme_jobs"] = {"url": "http://127.0.0.1", "enabled": False}
                    if jobs:
                        url = await self._mcp_url()
                        if state.token is None:
                            state.token = self.jobs_mcp.register(jobs, injected)
                            created_token = state.token
                        servers["reme_jobs"] = {
                            "url": url,
                            "http_headers": {"Authorization": f"Bearer {state.token}"},
                            "required": True,
                            "enabled_tools": [job.name for job in jobs],
                            "startup_timeout_sec": mcp_startup_timeout,
                            "tool_timeout_sec": mcp_tool_timeout,
                        }
                    config["mcp_servers"] = servers
                    options["config"] = config
                if fork:
                    source = self._threads.setdefault(thread_id, _ThreadState())
                    source.users += 1
                    source.revision += 1
                    try:
                        async with source.lock:
                            thread = await codex.thread_fork(thread_id, **options)
                    finally:
                        source.users -= 1
                elif thread_id:
                    thread = await codex.thread_resume(thread_id, **options)
                else:
                    thread = await codex.thread_start(**options)
                self._threads[thread.id] = state
                if thread_id and not fork:
                    await self.wait_idle(thread)
                try:
                    yield thread
                finally:
                    # Interrupt acknowledgement alone does not mean that tool execution stopped.
                    error = sys.exception()
                    try:
                        async with asyncio.timeout(30 if error is not None else None):
                            await self.wait_idle(thread)
                    except Exception as exc:
                        if isinstance(exc, TransportClosedError) and self._codex is codex:
                            self._failed = True
                        if error is None:
                            raise
                        self.logger.warning(f"Failed to wait for Codex thread {thread.id}: {exc}")
                    else:
                        from openai_codex.generated.v2_all import ThreadUnsubscribeResponse

                        try:
                            await codex._client.request(  # pylint: disable=protected-access
                                "thread/unsubscribe",
                                {"threadId": thread.id},
                                response_model=ThreadUnsubscribeResponse,
                            )
                        except Exception as exc:
                            self.logger.warning(f"Failed to unsubscribe Codex thread {thread.id}: {exc}")
        except TransportClosedError:
            if codex is not None and self._codex is codex:
                self._failed = True
            raise
        finally:
            if state is not None:
                state.users -= 1
                if thread is None and created_token is not None:
                    self.jobs_mcp.scopes.pop(created_token, None)
                    state.token = None
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def _close(self) -> None:
        await self._idle.wait()
        try:
            await self._close_client()
        finally:
            self._codex = None
            self._failed = False
            self._http_url = None
            self._threads.clear()
            await self.jobs_mcp.close()
