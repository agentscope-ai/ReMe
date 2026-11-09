"""Application-owned Codex process, concurrent sessions, and local job tools."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass, field
import json
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, TYPE_CHECKING

from pydantic_core import to_jsonable_python

from .base_component import BaseComponent
from .component_registry import R
from .service.scoped_mcp import ScopedMCP
from ..constants import normalize_connect_host
from ..config.config_parser import deep_merge_config
from ..enumeration import ComponentEnum
from ..utils.async_utils import finish_cleanup
from ..utils.async_utils import complete_in_thread
from ..schema import AgentActivity, SessionMessage, SessionSnapshot, SessionStatus, TokenUsage

if TYPE_CHECKING:
    from openai_codex import AsyncCodex, AsyncThread, CodexConfig


class SessionChangedError(RuntimeError):
    """A reminder's conversation anchor no longer matches the target session."""


@dataclass
class _ThreadState:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    revision: int = 0
    binding: str | None = None
    token: str | None = None
    completion: asyncio.Future[None] | None = None
    operation_started: bool = False
    token_usage: TokenUsage | None = None
    pending: AsyncThread | None = None
    history_changed: bool = False
    options: dict[str, Any] = field(default_factory=dict)
    turn_options: dict[str, Any] = field(default_factory=dict)


@R.register("sdk")
class CodexAppServer(BaseComponent):
    """Share one SDK client across wrappers with the same account and launch settings."""

    component_type = ComponentEnum.CODEX_APP_SERVER

    def __init__(
        self,
        auth_mode: str = "auto",
        api_key: str = "",
        base_url: str = "",
        supports_websockets: bool = False,
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
        self._supports_websockets = supports_websockets
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
        self._session_listeners: dict[
            str,
            set[Callable[[AgentActivity | SessionMessage | SessionSnapshot | SessionStatus | None], None]],
        ] = {}
        self._activity_listeners: set[Callable[[AgentActivity | None], None]] = set()

    def watch_activity(self, listener: Callable[[AgentActivity | None], None]) -> Callable[[], None]:
        """Observe execution across native sessions; None invalidates all activity."""
        self._activity_listeners.add(listener)
        return lambda: self._activity_listeners.discard(listener)

    def watch_session(
        self,
        session_id: str,
        listener: Callable[[AgentActivity | SessionMessage | SessionSnapshot | SessionStatus | None], None],
    ) -> Callable[[], None]:
        """Observe one conversation on this event loop; None invalidates cached state."""
        self._message_path(session_id)
        self._session_listeners.setdefault(session_id, set()).add(listener)

        def unsubscribe():
            listeners = self._session_listeners.get(session_id)
            if listeners is not None:
                listeners.discard(listener)
                if not listeners:
                    self._session_listeners.pop(session_id, None)

        return unsubscribe

    def _notify_session(
        self,
        session_id: str | None,
        change: AgentActivity | SessionMessage | SessionSnapshot | SessionStatus | None,
    ) -> None:
        if session_id is None and change is None:
            self._notify_activity(None)
        listeners = (
            set(self._session_listeners.get(session_id, ()))
            if session_id is not None
            else {listener for group in self._session_listeners.values() for listener in group}
        )
        for listener in listeners:
            try:
                listener(change)
            except Exception as exc:
                self.logger.warning(f"Session listener failed: {exc}")

    def publish_activity(self, activity: AgentActivity) -> None:
        """Publish a phase without coupling model execution to desktop observers."""
        self._notify_session(activity.session_id, activity)
        self._notify_activity(activity)

    def _notify_activity(self, activity: AgentActivity | None) -> None:
        for listener in tuple(self._activity_listeners):
            try:
                listener(activity)
            except Exception as exc:
                self.logger.warning(f"Activity listener failed: {exc}")

    async def _publish_history(self, session_id: str) -> None:
        """Confirm visible history after a turn, without coupling observers to its outcome."""
        if session_id not in self._session_listeners:
            return
        try:
            async with asyncio.timeout(30):
                snapshot = await self.read_session(session_id)
        except Exception as exc:
            self.logger.warning(f"Failed to refresh published session {session_id}: {exc}")
            self._notify_session(session_id, None)
        else:
            self._notify_session(session_id, snapshot)

    @property
    def session_path(self) -> Path:
        """Persistent Codex state under the workspace, unless an explicit home is configured."""
        if self._codex_home:
            path = Path(self._codex_home).expanduser()
            return path if path.is_absolute() else self.workspace_path / path
        return self.workspace_path / ".codex"

    def _session_file_path(self, thread: Any) -> Path | None:
        """Validate a native path without requiring its file to exist yet."""
        if not (raw_path := getattr(thread, "path", None)):
            return None
        path = Path(raw_path).resolve()
        if not path.is_relative_to(self.session_path.resolve()):
            raise ValueError("Codex session path is outside its configured home")
        return path

    def _message_path(self, session_id: str) -> Path:
        from .agent_wrapper.codex_session_history import validate_identifier

        validate_identifier(session_id)
        session_dir = self.app_context.app_config.session_dir if self.app_context else "session"
        namespace = hashlib.sha256(str(self.session_path.resolve()).encode()).hexdigest()[:16]
        path = (self.workspace_path / session_dir / "proactive" / namespace / f"{session_id}.jsonl").resolve()
        if not path.is_relative_to(self.workspace_path.resolve()):
            raise ValueError("Session messages must stay inside the workspace")
        return path

    @asynccontextmanager
    async def _session_client(self) -> AsyncIterator[AsyncCodex]:
        """Keep status/history RPCs under lifecycle ownership without taking the turn lock."""
        await self.start()
        self._active += 1
        self._idle.clear()
        try:
            yield await self._get_client()
        finally:
            self._active -= 1
            if not self._active:
                self._idle.set()

    async def release_session(self, session_id: str) -> None:
        """Unsubscribe a superseded empty thread without deleting any session files."""
        state = self._threads.get(session_id)
        if state is not None and state.pending is not None:
            async with self.thread(session_id):
                state.pending = None

    async def get_session_status(self, session_id: str) -> SessionStatus:
        """Inspect a thread without resuming it or waiting for an active turn."""
        from .agent_wrapper.codex_session_history import session_status

        self._message_path(session_id)
        async with self._session_client() as codex:
            thread = (await codex._client.thread_read(session_id)).thread  # pylint: disable=protected-access
            return session_status(thread)

    async def get_session_path(self, session_id: str) -> Path | None:
        """Read only native metadata; the reported file may not have been created yet."""
        from .agent_wrapper.codex_session_history import validate_identifier

        validate_identifier(session_id)
        async with self._session_client() as codex:
            response = await codex._client.thread_read(  # pylint: disable=protected-access
                session_id,
                include_turns=False,
            )
            return self._session_file_path(response.thread)

    async def read_session(self, session_id: str) -> SessionSnapshot:
        """Project public native history plus application-authored proactive messages."""
        from .agent_wrapper import codex_session_history as history

        path = self._message_path(session_id)
        async with self._session_client() as codex:
            state = self._threads.get(session_id)
            response = await codex._client.thread_read(  # pylint: disable=protected-access
                session_id,
                include_turns=state is None or state.pending is None,
            )
            thread = response.thread
            records = await asyncio.to_thread(history.read_records, path)
            recovered = (
                await asyncio.to_thread(history.injected_ids, thread, self.session_path)
                if any(not record["injected"] for record in records)
                else set()
            )
            snapshot = history.session_snapshot(thread, records, recovered)
            session_file = self._session_file_path(thread)
            snapshot.path = str(session_file) if session_file is not None else None
            return snapshot

    async def append_session(
        self,
        session_id: str,
        content: str,
        *,
        message_id: str,
        timeout: float = 30,
        expected_turn_id: str | None = None,
        delivery: str = "idle",
    ) -> SessionMessage:
        """Wait for the thread lease, then inject and publish one retry-safe assistant message."""
        from openai_codex.generated.v2_all import ThreadInjectItemsResponse
        from .agent_wrapper import codex_session_history as history

        path = self._message_path(session_id)
        if delivery not in {"idle", "immediate"}:
            raise ValueError("Invalid chat delivery policy")
        history.validate_identifier(message_id)
        if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
            raise ValueError("content must contain 1-100000 characters")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")

        async with asyncio.timeout(timeout):
            async with self.thread(session_id) as loaded:
                thread = (await loaded.read(include_turns=self._threads[session_id].pending is None)).thread
                if history.session_status(thread).status != "idle":
                    raise SessionChangedError("Session must be idle before appending")
                accepted = await asyncio.to_thread(history.injected_ids, thread, self.session_path)
                records = await asyncio.to_thread(history.read_records, path)
                native_id = f"reme-proactive-{message_id}"
                record = next((entry for entry in records if entry["message"]["id"] == native_id), None)
                if record is not None:
                    if record["message"]["content"] != content:
                        raise ValueError("message_id already belongs to different content")
                    if record["injected"] or native_id in accepted:

                        async def recover_and_publish():
                            if not record["injected"]:
                                record["injected"] = True
                                await complete_in_thread(history.write_records, path, records)
                            message = SessionMessage.model_validate(record["message"])
                            self._notify_session(session_id, message)
                            return message

                        return await finish_cleanup(recover_and_publish())
                if expected_turn_id is not None and (thread.turns[-1].id if thread.turns else "") != expected_turn_id:
                    raise SessionChangedError("Session changed before appending")
                if record is None:
                    record = history.new_record(message_id, content, thread)
                    record["message"].update(delivery=delivery, origin_session_id=session_id)
                    records.append(record)
                # An unaccepted attempt may be retried after later user turns.
                record["message"]["turn_id"] = thread.turns[-1].id if thread.turns else None

                async def inject_and_publish():
                    # A crash between the RPC and publication leaves a pending record.
                    # Its reserved native ID is recovered from the rollout on retry.
                    await complete_in_thread(history.write_records, path, records)
                    codex = await self._get_client()
                    await codex._client.request(  # pylint: disable=protected-access
                        "thread/inject_items",
                        {
                            "threadId": session_id,
                            "items": [
                                {
                                    "type": "message",
                                    "id": native_id,
                                    "role": "assistant",
                                    "content": [{"type": "output_text", "text": content}],
                                },
                            ],
                        },
                        response_model=ThreadInjectItemsResponse,
                    )
                    record["injected"] = True
                    await complete_in_thread(history.write_records, path, records)
                    self._notify_session(session_id, SessionMessage.model_validate(record["message"]))

                await finish_cleanup(inject_and_publish())
                return SessionMessage.model_validate(record["message"])

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
            # Custom endpoints must not inherit the built-in OpenAI provider's WebSocket capability.
            overrides = [
                'model_provider="reme"',
                'model_providers.reme.name="ReMe API"',
                f"model_providers.reme.base_url={json.dumps(self._base_url)}",
                'model_providers.reme.wire_api="responses"',
                "model_providers.reme.requires_openai_auth=true",
                f"model_providers.reme.supports_websockets={json.dumps(self._supports_websockets)}",
                *overrides,
            ]
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
                native = event.payload.status.root
                self._notify_session(
                    event.payload.thread_id,
                    SessionStatus(
                        session_id=event.payload.thread_id,
                        status={"notLoaded": "not_loaded", "systemError": "error"}.get(native.type, native.type),
                        active_flags=[getattr(flag, "value", flag) for flag in getattr(native, "active_flags", [])],
                    ),
                )
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
                self._notify_session(None, None)
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
        """Track a shielded dispatch; clear its barrier if the SDK rejects the request."""
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

    async def token_usage(self, thread_id: str) -> TokenUsage:
        """Read the pre-turn baseline under the conversation lease, recovering it after restart."""
        from .agent_wrapper.codex_session_history import read_token_usage

        state = self._threads[thread_id]
        if state.token_usage is None:
            codex = await self._get_client()
            view = (await codex._client.thread_read(thread_id)).thread  # pylint: disable=protected-access
            state.token_usage = await asyncio.to_thread(read_token_usage, view, self.session_path)
        return state.token_usage

    def update_token_usage(self, thread_id: str, total: TokenUsage) -> TokenUsage:
        """Keep cumulative provider counters monotonic across wrappers and duplicate notifications."""
        state = self._threads[thread_id]
        previous = state.token_usage or TokenUsage()
        state.token_usage = TokenUsage(
            input_tokens=max(previous.input_tokens, total.input_tokens),
            output_tokens=max(previous.output_tokens, total.output_tokens),
        )
        return state.token_usage

    async def turn(self, thread: AsyncThread, inputs: Any, **kwargs):
        """Start a normal SDK turn with completion tracking owned by this server."""
        turn = None

        async def dispatch():
            nonlocal turn
            state = self._threads[thread.id]
            options = {**state.turn_options, **{key: value for key, value in kwargs.items() if value is not None}}
            async with self._operation(thread):
                turn = await thread.turn(inputs, **options)
                state.pending = None
                state.history_changed = True
                self._notify_session(thread.id, SessionStatus(session_id=thread.id, status="active"))
            state.turn_options = options
            await self._save_thread_config(thread.id, state)

        try:
            await finish_cleanup(dispatch())
        except asyncio.CancelledError:
            if turn is not None:
                try:
                    await finish_cleanup(turn.interrupt())
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    self.logger.warning(f"Failed to interrupt Codex turn {turn.id}: {exc}")
            raise
        return turn

    async def compact(self, thread: AsyncThread) -> None:
        """Enqueue native compaction; the thread lease waits for its completion event."""

        async def dispatch():
            async with self._operation(thread):
                await thread.compact()

        self._threads[thread.id].token_usage = None
        await finish_cleanup(dispatch())

    async def _fork_messages(self, source_id: str, thread: AsyncThread) -> None:
        """Copy inherited proactive provenance while the source conversation is locked."""
        from .agent_wrapper import codex_session_history as history

        records = await asyncio.to_thread(history.read_records, self._message_path(source_id))
        if not records:
            return
        accepted = set()
        if any(not record["injected"] for record in records):
            codex = await self._get_client()
            source = (await codex._client.thread_read(source_id)).thread  # pylint: disable=protected-access
            accepted = await asyncio.to_thread(history.injected_ids, source, self.session_path)
        # Native forks inherit context but do not preserve injected item IDs.
        inherited = [
            {**record, "injected": True}
            for record in records
            if record["injected"] or record["message"]["id"] in accepted
        ]
        if inherited:
            await complete_in_thread(history.write_records, self._message_path(thread.id), inherited)

    async def _save_thread_config(self, thread_id: str, state: _ThreadState) -> None:
        """Retain effective settings without persisting generated MCP credentials."""
        from .agent_wrapper.codex_session_history import write_records

        path = self._message_path(thread_id).with_suffix(".config.jsonl")
        await complete_in_thread(
            write_records,
            path,
            [
                to_jsonable_python(
                    {"options": state.options, "turn_options": state.turn_options, "binding": state.binding},
                ),
            ],
        )

    @staticmethod
    def _native_options(options):
        """Restore SDK option types from the file-native session settings."""
        from openai_codex import ApprovalMode, Sandbox
        from openai_codex.types import Personality, ReasoningEffort, ReasoningSummary, ThreadSource, ThreadStartSource

        types = {
            "approval_mode": ApprovalMode,
            "sandbox": Sandbox,
            "personality": Personality,
            "effort": ReasoningEffort,
            "summary": ReasoningSummary,
            "thread_source": ThreadSource,
            "session_start_source": ThreadStartSource,
        }
        return {
            key: types[key](value) if key in types and value is not None else value for key, value in options.items()
        }

    async def _restore_thread_config(self, thread_id, state, *, restore_binding=True):
        """Load a prior configuration without reviving expired transport credentials."""
        from .agent_wrapper.codex_session_history import read_records

        if state.options:
            return
        records = await asyncio.to_thread(read_records, self._message_path(thread_id).with_suffix(".config.jsonl"))
        if records:
            if len(records) != 1:
                raise ValueError("Invalid Codex session configuration")
            settings = records[0]
            state.options = self._native_options(settings["options"])
            state.turn_options = self._native_options(settings["turn_options"])
            if restore_binding:
                state.binding = settings["binding"]

    @asynccontextmanager
    async def _fork_source(self, thread_id: str) -> AsyncIterator[_ThreadState]:
        """Pin a completed source and restore settings after unload or application restart."""
        source = self._threads.setdefault(thread_id, _ThreadState())
        source.users += 1
        source.revision += 1
        try:
            async with source.lock:
                await self._restore_thread_config(thread_id, source)
                yield source
        finally:
            source.users -= 1

    async def _bind_tools(self, state, options, jobs, injected, startup_timeout, tool_timeout):
        """Rebind transport credentials while retaining the model-visible tool catalog."""
        if jobs is None:
            return options
        injected = json.loads(json.dumps(injected or {}))
        binding = json.dumps({"jobs": [job.name for job in jobs], "injected": injected}, sort_keys=True)
        if state.binding is not None and state.binding != binding:
            raise ValueError("job_tools, tool_context_id and injected_job_kwargs cannot change on a loaded thread")
        state.binding = binding
        config = dict(options.get("config") or {})
        servers = dict(config.get("mcp_servers") or {})
        if "reme_jobs" in servers:
            raise ValueError("mcp_servers.reme_jobs is reserved for ReMe job tools")
        servers["reme_jobs"] = {"url": "http://127.0.0.1", "enabled": False}
        if jobs:
            url = await self._mcp_url()
            if state.token is None:
                state.token = self.jobs_mcp.register(jobs, injected)
            servers["reme_jobs"] = {
                "url": url,
                "http_headers": {"Authorization": f"Bearer {state.token}"},
                "required": True,
                "enabled_tools": [job.name for job in jobs],
                "tools": {job.name: {"approval_mode": "approve"} for job in jobs},
                "startup_timeout_sec": startup_timeout,
                "tool_timeout_sec": tool_timeout,
            }
        return {**options, "config": {**config, "mcp_servers": servers}}

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
        keep_empty: bool = False,
    ) -> AsyncIterator[AsyncThread]:
        """Serialize one conversation while allowing independent conversations to run concurrently."""
        from openai_codex import TransportClosedError

        await self.start()
        self._active += 1
        self._idle.clear()
        state = None
        previous_token = None
        thread = None
        codex = None
        try:
            codex = await self._get_client()
            await self._prune_threads(codex)
            state = self._threads.setdefault(thread_id, _ThreadState()) if thread_id and not fork else _ThreadState()
            state.users += 1
            state.revision += 1
            previous_token = state.token
            async with state.lock:
                if thread_id and not fork:
                    await self._restore_thread_config(thread_id, state, restore_binding=jobs is None)
                async with self._fork_source(thread_id) if fork else nullcontext(None) as source:
                    options = dict(options or {})
                    if source is not None:
                        inherited = dict(source.options)
                        state.turn_options = dict(source.turn_options)
                        for key in ("cwd", "model", "approval_mode", "sandbox", "service_tier"):
                            if key in source.turn_options:
                                inherited[key] = source.turn_options[key]
                        options = deep_merge_config(inherited, options)
                        personality = options.pop("personality", None)
                        if personality is not None:
                            options["config"] = {
                                **options.get("config", {}),
                                "personality": to_jsonable_python(personality),
                            }
                        for key in ("service_name", "session_start_source"):
                            options.pop(key, None)
                        if jobs is None and source.binding is not None:
                            binding = json.loads(source.binding)
                            jobs = [self.app_context.jobs[name] for name in binding["jobs"]]
                            injected = {**binding["injected"], **(injected or {})}
                    effective = {key: value for key, value in options.items() if value is not None}
                    state.options = deep_merge_config(state.options, effective)
                    options = await self._bind_tools(
                        state,
                        options,
                        jobs,
                        injected,
                        mcp_startup_timeout,
                        mcp_tool_timeout,
                    )
                    if fork:
                        # Native fork requires a rollout, which an unstarted live thread may not have yet.
                        source_path = await self.get_session_path(thread_id) if source.pending is not None else None
                        empty = source.pending is not None and (
                            source_path is None or not await asyncio.to_thread(source_path.exists)
                        )
                        if empty:
                            thread = await codex.thread_start(**options)
                            state.token_usage = TokenUsage()
                        else:
                            state.token_usage = await self.token_usage(thread_id)
                            thread = await codex.thread_fork(thread_id, **options)
                            await finish_cleanup(self._fork_messages(thread_id, thread))
                    elif state.pending is not None:
                        thread = state.pending
                    elif thread_id:
                        thread = await codex.thread_resume(thread_id, **options)
                    else:
                        thread = await codex.thread_start(**options)
                        state.token_usage = TokenUsage()
                        if keep_empty:
                            state.pending = thread
                self._threads[thread.id] = state
                if thread_id and not fork:
                    await self.wait_idle(thread)
                state.history_changed = False
                try:
                    await self._save_thread_config(thread.id, state)
                    yield thread
                finally:
                    # Interrupt acknowledgement alone does not mean that tool execution stopped.
                    error = sys.exception()
                    if error is not None and state.pending is None:
                        # A paused/closed consumer can miss usage emitted by native completion.
                        # Recover from the persisted rollout before the next turn starts.
                        state.token_usage = None
                    try:
                        async with asyncio.timeout(30 if error is not None else None):
                            await self.wait_idle(thread)
                    except Exception as exc:
                        if isinstance(exc, TransportClosedError) and self._codex is codex:
                            self._failed = True
                        self._notify_session(thread.id, None)
                        if error is None:
                            raise
                        self.logger.warning(f"Failed to wait for Codex thread {thread.id}: {exc}")
                    else:
                        if state.history_changed:
                            await finish_cleanup(self._publish_history(thread.id))
                        from openai_codex.generated.v2_all import ThreadUnsubscribeResponse

                        try:
                            if state.pending is None:
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
                self._notify_session(None, None)
            raise
        finally:
            if state is not None:
                state.users -= 1
                if thread is None and state.token != previous_token:
                    self.jobs_mcp.scopes.pop(state.token, None)
                    state.token = previous_token
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
            self._session_listeners.clear()
            await self.jobs_mcp.close()
