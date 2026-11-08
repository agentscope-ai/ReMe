"""Collection-side plumbing shared by credentialed remote data sources."""

import asyncio
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import TypeVar
from uuid import uuid4

from .base_connector import BaseConnectorStep

T = TypeVar("T")


class NothingToPublish(RuntimeError):
    """Raised inside ``collect`` when a poll produced no payload, so no package is published."""


class MissingCredentials(RuntimeError):
    """Raised when required credentials are absent, so the source cannot be polled at all."""


class TransientFetchError(RuntimeError):
    """A retryable provider or transport error, optionally carrying an advised wait in seconds."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class BaseRemoteConnectorStep(BaseConnectorStep):
    """Add credentials, cursors, throttling, retries, and pagination on top of ``collect``.

    Subclasses still write every payload inside :meth:`collect`. Cursor updates queued with
    :meth:`stage_state` are persisted only after the package has been published atomically, so a
    failed or cancelled poll re-reads the same window instead of silently dropping items. A poll
    without findings raises :class:`NothingToPublish` and publishes nothing.
    """

    #: Environment variable consulted for a setting that YAML and the caller both omit.
    env_defaults: dict[str, str] = {}
    default_timeout_seconds = 30.0
    default_max_retries = 3
    default_backoff_seconds = 2.0
    max_backoff_seconds = 300.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loaded_state: dict | None = None
        self._pending_state: dict = {}

    # ----- settings and credentials -------------------------------------------------------

    def setting(self, key: str, default=None):
        """Return the step kwarg ``key``, then its mapped environment variable, then ``default``."""
        for value in (self.kwargs.get(key), os.environ.get(self.env_defaults.get(key, ""))):
            if value is not None and value != "":
                return value
        return default

    def text_setting(self, key: str, default: str = "") -> str:
        """Return a setting as a stripped string."""
        return str(self.setting(key, default)).strip()

    def int_setting(self, key: str, default: int, *, minimum: int = 0) -> int:
        """Return a setting as an integer no smaller than ``minimum``."""
        value = self.setting(key, default)
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer, got {value!r}") from exc
        if number < minimum:
            raise ValueError(f"{key} must be at least {minimum}, got {number}")
        return number

    def float_setting(self, key: str, default: float, *, minimum: float = 0.0) -> float:
        """Return a setting as a finite float no smaller than ``minimum``."""
        value = self.setting(key, default)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number, got {value!r}") from exc
        if not math.isfinite(number) or number < minimum:
            raise ValueError(f"{key} must be a finite number of at least {minimum}, got {number}")
        return number

    def bool_setting(self, key: str, default: bool = False) -> bool:
        """Return a setting as a boolean, accepting YAML booleans and common text forms."""
        value = self.setting(key, default)
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"", "0", "false", "no", "off"}:
            return False
        raise ValueError(f"{key} must be a boolean, got {value!r}")

    def list_setting(self, key: str, default: Sequence[str] = ()) -> list[str]:
        """Return a setting as a list of strings, accepting YAML lists and delimited text."""
        value = self.setting(key, None)
        if value is None:
            return [str(item) for item in default]
        if isinstance(value, str):
            items = [item.strip() for item in re.split(r"[,\n]+", value)]
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value]
        else:
            raise ValueError(f"{key} must be a list of strings, got {type(value).__name__}")
        return [item for item in items if item]

    def credential(self, key: str) -> str:
        """Return one required secret without ever logging it, naming its source when absent."""
        value = self.setting(key)
        if value is None or not str(value).strip():
            raise MissingCredentials(f"missing credential {key!r}; set {self.env_defaults.get(key, key)}")
        return str(value).strip()

    def timeout_seconds(self) -> float:
        """Return the deadline applied to every network call in one poll."""
        return self.float_setting("timeout_seconds", self.default_timeout_seconds, minimum=0.1)

    # ----- retries and throttling ---------------------------------------------------------

    def retryable(self, exc: Exception) -> bool:
        """Whether a fetch error is transient and worth another attempt; subclasses extend this."""
        return isinstance(exc, (TransientFetchError, TimeoutError, ConnectionError))

    def retry(self, operation: Callable[[], T], *, description: str) -> T:
        """Run one blocking fetch with bounded exponential backoff and jitter."""
        attempts = self.int_setting("max_retries", self.default_max_retries) + 1
        backoff = self.float_setting("retry_backoff_seconds", self.default_backoff_seconds, minimum=0.1)
        for attempt in range(attempts):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 - classification belongs to retryable()
                if not self.retryable(exc) or attempt + 1 >= attempts:
                    raise
                advised = exc.retry_after if isinstance(exc, TransientFetchError) else None
                wait = min(self.max_backoff_seconds, max(advised or 0.0, backoff * 2**attempt))
                wait += random.uniform(0, min(1.0, wait * 0.25))
                self.logger.warning(f"{description} failed ({exc}); retrying in {wait:.2f}s")
                self._sleep(wait)
        raise RuntimeError(f"{description} failed after {attempts} attempts")

    def _sleep(self, seconds: float) -> None:
        """Wait inside the collector thread, giving up as soon as a stop is requested."""
        stop = self.context.stop_event if self.context is not None else None
        if isinstance(stop, Event) and stop.is_set():
            raise RuntimeError("Resource collection stopped before a retry")
        time.sleep(max(0.0, seconds))

    def throttled(self) -> float | None:
        """Return the minutes left to wait when another poll would come too soon, else ``None``."""
        minutes = self.float_setting("min_interval_minutes", 0.0)
        if minutes <= 0:
            return None
        previous = self.parse_iso(self.load_state().get("last_poll_at"))
        if previous is None:
            return None
        elapsed = (self.utc_now() - previous).total_seconds() / 60
        return max(0.0, minutes - elapsed) if elapsed < minutes else None

    # ----- cursor state -------------------------------------------------------------------

    @property
    def state_key(self) -> str:
        """Identity of this collector's cursor file; override to run several accounts side by side."""
        return str(self.kwargs.get("state_key") or self.backend or type(self).__name__)

    def state_path(self) -> Path:
        """Return the workspace metadata file holding this collector's cursor."""
        return self.workspace_path / str(self.config_value("metadata_dir")) / "step" / f"{self.state_key}.json"

    def load_state(self) -> dict:
        """Return the persisted cursor state, or an empty mapping on the first run."""
        if self._loaded_state is None:
            path = self.state_path()
            state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            self._loaded_state = state if isinstance(state, dict) else {}
        return self._loaded_state

    def cursor(self) -> dict:
        """Return this collector's cursor mapping, which only it interprets."""
        return dict(self.load_state().get("cursor") or {})

    def stage_state(self, **updates) -> None:
        """Queue cursor fields to persist once this poll's package has been published."""
        self._pending_state.update(updates)

    def _commit_state(self, now: datetime | None = None) -> None:
        """Persist queued cursor fields atomically; an empty poll only refreshes poll bookkeeping."""
        pending, self._pending_state = self._pending_state, {}
        merged = {**self.load_state(), **pending}
        merged.update(
            version=1,
            connector=self.state_key,
            source=self.source,
            last_poll_at=self.iso(now or self.utc_now()),
            polls=int(merged.get("polls") or 0) + 1,
        )
        path = self.state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        self._loaded_state = merged

    # ----- publication hooks --------------------------------------------------------------

    def describe(self, description: str) -> None:
        """Refine this package's metadata description from inside ``collect``.

        ``BaseConnectorStep`` reads the ``description`` kwarg after ``collect`` returns, so a
        poll-specific summary reaches ``metadata.json`` without changing the shared base class.
        """
        self.kwargs["description"] = str(description).strip()

    def _collect_and_publish(self, target: Path, created_at: datetime, cancelled: Event) -> dict:
        metadata = super()._collect_and_publish(target, created_at, cancelled)
        self._commit_state(now=created_at)
        return metadata

    async def execute(self):
        """Poll once, turning absent credentials and empty results into a successful no-package run."""
        try:
            remaining = await asyncio.to_thread(self.throttled)
            if remaining is not None:
                # Refusing a poll must not refresh last_poll_at, which would extend this interval.
                return self._skipped(f"polled {remaining:.1f} minutes ago; min_interval_minutes has not elapsed")
            return await super().execute()
        except MissingCredentials as exc:
            self.logger.warning(f"{self.state_key} collected nothing: {exc}")
            return self._skipped(f"missing_credentials: {exc}")
        except NothingToPublish as exc:
            self.logger.info(f"{self.state_key} collected nothing: {exc}")
            await asyncio.to_thread(self._commit_state)
            return self._skipped(str(exc))

    def _skipped(self, reason: str):
        """Report a successful poll that published no package, keeping the Job result inspectable."""
        assert self.context is not None
        result = {"path": None, "files": [], "skipped": reason}
        self.context.response.success = True
        self.context.response.answer = result
        self.context.response.metadata.update(result)
        return self.context.response

    # ----- payload helpers ----------------------------------------------------------------

    def write_payload(self, directory: Path, filename: str, text: str) -> dict:
        """Write one UTF-8 payload inside the staging directory and return its file entry."""
        cleaned = str(filename).replace("\\", "/").strip("/")
        parts = [part for part in cleaned.split("/") if part != "."]
        if not parts or any(part == ".." or part.startswith("~") for part in parts):
            raise ValueError(f"Refusing to publish an unsafe payload path: {filename!r}")
        parts = [self.safe_name(part) for part in parts]
        target = directory.joinpath(*parts)
        content = str(text).strip() + "\n"
        if not content.strip():
            raise ValueError(f"Refusing to publish an empty payload: {filename!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": "/".join(parts)}

    @staticmethod
    def safe_name(value: str, *, limit: int = 64, fallback: str = "item") -> str:
        """Reduce untrusted provider text to one portable filename component."""
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
        return cleaned[:limit].strip("._-") or fallback

    @staticmethod
    def truncate(text: str, limit: int) -> str:
        """Cut text to ``limit`` UTF-8 bytes on a character boundary and mark the cut."""
        if limit <= 0:
            return ""
        encoded = str(text).encode("utf-8")
        if len(encoded) <= limit:
            return str(text)
        clipped = encoded[:limit].decode("utf-8", "ignore")
        return f"{clipped}\n\n[truncated {len(encoded) - len(clipped.encode('utf-8'))} bytes]"

    @staticmethod
    def utc_now() -> datetime:
        """Return the current time as an aware UTC datetime."""
        return datetime.now(timezone.utc)

    @staticmethod
    def iso(moment: datetime) -> str:
        """Render an aware datetime in the ``...Z`` form REST cursors expect."""
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def parse_iso(value) -> datetime | None:
        """Parse an ISO-8601 or HTTP-date cursor into an aware datetime, or ``None``."""
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            from email.utils import parsedate_to_datetime

            try:
                moment = parsedate_to_datetime(text)
            except (TypeError, ValueError):
                return None
        return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
