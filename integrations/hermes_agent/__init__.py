"""Hermes Agent memory provider backed by a running ReMe HTTP service."""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import re
import tempfile
import threading
import time

from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

from .client import ReMeHttpClient, ReMeServiceError

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "reme.json"
_DEFAULT_CONFIG: dict[str, Any] = {
    "endpoint": "http://127.0.0.1:2333",
    "request_timeout": 600.0,
    "recall_timeout": 5.0,
    "health_timeout": 2.0,
    "health_retry_seconds": 30.0,
    "shutdown_timeout": 30.0,
    "recall_limit": 5,
}
_NON_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _config_path(hermes_home: str | Path | None = None) -> Path:
    if hermes_home is None:
        from hermes_constants import get_hermes_home

        hermes_home = get_hermes_home()
    return Path(hermes_home).expanduser() / _CONFIG_FILENAME


def _load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    config = dict(_DEFAULT_CONFIG)
    path = _config_path(hermes_home)
    if not path.is_file():
        return config
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Unable to read ReMe provider config %s: %s", path, exc)
        return config
    if isinstance(loaded, dict):
        config.update({key: value for key, value in loaded.items() if value is not None and value != ""})
    return config


def _positive_float(config: dict[str, Any], key: str) -> float:
    try:
        return max(0.1, float(config[key]))
    except (KeyError, TypeError, ValueError):
        return float(_DEFAULT_CONFIG[key])


def _positive_int(config: dict[str, Any], key: str) -> int:
    try:
        return max(1, int(config[key]))
    except (KeyError, TypeError, ValueError):
        return int(_DEFAULT_CONFIG[key])


def _slug(value: str, fallback: str, *, limit: int) -> str:
    value = _NON_FILENAME_CHARS.sub("-", str(value or "").strip()).strip("-._")
    return (value or fallback)[:limit]


def _scoped_session_id(profile_id: str, session_id: str) -> str:
    """Create a readable, filename-safe ID without allowing scope collisions."""
    profile = str(profile_id or "default")
    session = str(session_id or "session")
    digest = hashlib.sha256(f"{profile}\0{session}".encode("utf-8")).hexdigest()[:12]
    return f"hermes-{_slug(profile, 'default', limit=32)}-{_slug(session, 'session', limit=64)}-{digest}"


class ReMeMemoryProvider(MemoryProvider):
    """Use ReMe for automatic cross-session recall and recording in Hermes."""

    def __init__(self) -> None:
        self._client: ReMeHttpClient | None = None
        self._endpoint = str(_DEFAULT_CONFIG["endpoint"])
        self._recall_timeout = float(_DEFAULT_CONFIG["recall_timeout"])
        self._health_timeout = float(_DEFAULT_CONFIG["health_timeout"])
        self._health_retry_seconds = float(_DEFAULT_CONFIG["health_retry_seconds"])
        self._shutdown_timeout = float(_DEFAULT_CONFIG["shutdown_timeout"])
        self._recall_limit = int(_DEFAULT_CONFIG["recall_limit"])
        self._service_available = False
        self._next_health_probe = 0.0
        self._next_recall_attempt = 0.0
        self._next_write_attempt = 0.0
        self._session_id = ""
        self._profile_id = "default"
        self._write_enabled = True
        self._accept_writes = True
        self._write_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._write_thread: threading.Thread | None = None
        self._write_thread_lock = threading.Lock()
        self._shutdown_started = False
        self._atexit_registered = False

    @property
    def name(self) -> str:
        """Return the provider identifier used by Hermes configuration."""
        return "reme"

    def is_available(self) -> bool:
        """Check local configuration only; network probes belong to initialize()."""
        try:
            config = _load_config()
            ReMeHttpClient(str(config["endpoint"]), timeout=_positive_float(config, "request_timeout"))
            return True
        except (KeyError, TypeError, ValueError, OSError):
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Load profile configuration and probe ReMe without blocking startup."""
        with self._write_thread_lock:
            if self._write_thread is not None and self._write_thread.is_alive():
                raise RuntimeError("Cannot reinitialize ReMe while its previous writer is still running")
        hermes_home = str(kwargs.get("hermes_home") or "") or None
        config = _load_config(hermes_home)
        self._endpoint = str(config["endpoint"])
        self._recall_timeout = _positive_float(config, "recall_timeout")
        self._health_timeout = _positive_float(config, "health_timeout")
        self._health_retry_seconds = _positive_float(config, "health_retry_seconds")
        self._shutdown_timeout = _positive_float(config, "shutdown_timeout")
        self._recall_limit = _positive_int(config, "recall_limit")
        self._session_id = str(session_id or "")
        self._profile_id = str(kwargs.get("agent_identity") or "default")
        self._write_enabled = str(kwargs.get("agent_context") or "primary") not in {"cron", "flush", "subagent"}
        self._client = ReMeHttpClient(self._endpoint, timeout=_positive_float(config, "request_timeout"))
        self._service_available = False
        self._next_health_probe = 0.0
        self._next_recall_attempt = 0.0
        self._next_write_attempt = 0.0
        self._accept_writes = True
        self._write_queue = queue.Queue()
        self._write_thread = None
        self._shutdown_started = False
        if not self._atexit_registered:
            atexit.register(self._atexit_shutdown)
            self._atexit_registered = True

        if not self._ensure_service(force=True):
            logger.warning(
                "ReMe is unavailable at %s; recall is disabled and completed "
                "turns will not be recorded until it recovers",
                self._endpoint,
            )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Describe interactive setup fields understood by Hermes."""
        return [
            {
                "key": "endpoint",
                "description": "ReMe HTTP service endpoint",
                "default": str(_DEFAULT_CONFIG["endpoint"]),
                "required": True,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Atomically save non-secret settings inside the active Hermes profile."""
        path = _config_path(hermes_home)
        existing = _load_config(hermes_home)
        existing.update({key: value for key, value in dict(values or {}).items() if value is not None and value != ""})

        # Validate before replacing a working configuration.
        client = ReMeHttpClient(
            str(existing["endpoint"]),
            timeout=_positive_float(existing, "request_timeout"),
        )
        client.health(timeout=_positive_float(existing, "health_timeout"))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(existing, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Automatic recall and capture add no model-visible tool schemas."""
        return []

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memory before Hermes sends a turn to the model."""
        del session_id
        query = str(query or "").strip()
        if not query or time.monotonic() < self._next_recall_attempt or not self._ensure_service():
            return ""
        assert self._client is not None
        try:
            response = self._client.call(
                "search",
                {"query": query, "limit": self._recall_limit},
                timeout=self._recall_timeout,
            )
        except ReMeServiceError as exc:
            self._next_recall_attempt = time.monotonic() + self._health_retry_seconds
            logger.warning("ReMe retrieval failed at %s: %s", self._endpoint, exc)
            return ""
        answer = response.get("answer")
        return answer.strip() if isinstance(answer, str) else ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Queue one completed turn without blocking Hermes on ReMe's LLM."""
        del messages
        user = str(user_content or "").strip()
        assistant = str(assistant_content or "").strip()
        if not self._write_enabled or not (user or assistant):
            return
        routed_session = str(session_id or self._session_id)
        if not routed_session:
            logger.warning("ReMe skipped a completed turn because Hermes supplied no session id")
            return
        if not self._accept_writes:
            logger.warning(
                "ReMe did not record completed turn for session %s because the provider is shutting down",
                _scoped_session_id(self._profile_id, routed_session),
            )
            return

        payload = {
            "session_id": _scoped_session_id(self._profile_id, routed_session),
            "messages": [
                {"name": "user", "role": "user", "content": user},
                {"name": "assistant", "role": "assistant", "content": assistant},
            ],
        }
        if not self._enqueue_write(payload):
            logger.warning(
                "ReMe did not record completed turn for session %s because the provider is shutting down",
                payload["session_id"],
            )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Update the active conversation boundary after a Hermes switch."""
        del parent_session_id, reset, rewound, kwargs
        if new_session_id:
            self._session_id = str(new_session_id)

    def shutdown(self) -> None:
        """Drain queued writes for a bounded interval, then release state."""
        with self._write_thread_lock:
            if self._shutdown_started:
                return
            self._shutdown_started = True
            self._accept_writes = False
            thread = self._write_thread
            if thread is not None:
                self._write_queue.put(None)
        if thread is not None:
            thread.join(timeout=self._shutdown_timeout)
            if thread.is_alive():
                abandoned = self._discard_queued_writes()
                logger.warning(
                    "ReMe shutdown timed out after %.1fs; abandoned %d queued write(s) "
                    "and the in-flight write may not finish before process exit",
                    self._shutdown_timeout,
                    abandoned,
                )
            else:
                self._client = None
        else:
            self._client = None
        self._service_available = False
        self._next_health_probe = 0.0

    def _atexit_shutdown(self) -> None:
        try:
            self.shutdown()
        except Exception as exc:  # pragma: no cover - interpreter teardown safety
            logger.debug("ReMe atexit shutdown failed: %s", exc)

    def _discard_queued_writes(self) -> int:
        abandoned = 0
        while True:
            try:
                payload = self._write_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if payload is not None:
                    abandoned += 1
            finally:
                self._write_queue.task_done()
        self._write_queue.put(None)
        return abandoned

    def _enqueue_write(self, payload: dict[str, Any]) -> bool:
        with self._write_thread_lock:
            if not self._accept_writes:
                return False
            if self._write_thread is None or not self._write_thread.is_alive():
                self._write_thread = threading.Thread(
                    target=self._write_loop,
                    args=(self._write_queue,),
                    daemon=True,
                    name="reme-memory-writer",
                )
                self._write_thread.start()
            self._write_queue.put(payload)
            return True

    def _write_loop(self, write_queue: queue.Queue[dict[str, Any] | None]) -> None:
        try:
            while True:
                payload = write_queue.get()
                try:
                    if payload is None:
                        return
                    try:
                        self._record_payload(payload)
                    except Exception as exc:  # keep one bad response from killing the writer
                        logger.exception(
                            "Unexpected ReMe recording failure for session %s; the writer will continue: %s",
                            payload.get("session_id", "<unknown>"),
                            exc,
                        )
                finally:
                    write_queue.task_done()
        finally:
            current_thread = threading.current_thread()
            with self._write_thread_lock:
                if self._write_thread is current_thread:
                    self._write_thread = None
                if not self._accept_writes:
                    self._client = None

    def _record_payload(self, payload: dict[str, Any]) -> None:
        if time.monotonic() < self._next_write_attempt:
            logger.warning(
                "ReMe did not record completed turn for session %s because writes are cooling down",
                payload["session_id"],
            )
            return
        if not self._ensure_service():
            logger.warning(
                "ReMe did not record completed turn for session %s because the service is unavailable",
                payload["session_id"],
            )
            return
        assert self._client is not None
        try:
            self._client.call("auto_memory", payload)
        except ReMeServiceError as exc:
            self._next_write_attempt = time.monotonic() + self._health_retry_seconds
            logger.warning("ReMe recording failed at %s: %s", self._endpoint, exc)
            logger.warning(
                "ReMe did not record completed turn for session %s",
                payload["session_id"],
            )

    def _ensure_service(self, *, force: bool = False) -> bool:
        if self._client is None:
            return False
        if self._service_available and not force:
            return True
        now = time.monotonic()
        if not force and now < self._next_health_probe:
            return False
        try:
            self._client.health(timeout=self._health_timeout)
        except ReMeServiceError as exc:
            self._mark_unavailable("health check", exc)
            return False
        self._service_available = True
        self._next_health_probe = 0.0
        return True

    def _mark_unavailable(self, operation: str, error: Exception) -> None:
        self._service_available = False
        self._next_health_probe = time.monotonic() + self._health_retry_seconds
        logger.warning("ReMe %s failed at %s: %s", operation, self._endpoint, error)


def register(ctx: Any) -> None:
    """Register with Hermes' memory-provider collector."""
    ctx.register_memory_provider(ReMeMemoryProvider())
