"""Lifecycle-owned daily consolidation using only the public ReMe MCP tools."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date as calendar_date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from reme_config import data_dir, load_config, lock, write_json
from reme_mcp import call_async
from reme_state import digest, endpoint_dir, log_status


def next_daily_run(cron: str, zone: str, now: datetime | None = None) -> datetime:
    """Skip nonexistent local times and run once on days with an ambiguous clock time."""
    now = now or datetime.now(timezone.utc)
    minute, hour = (int(value) for value in cron.split()[:2])
    tz = ZoneInfo(zone)
    day = now.astimezone(tz).date()
    for offset in range(5):
        candidate = datetime.combine(day + timedelta(days=offset), datetime.min.time()).replace(
            hour=hour,
            minute=minute,
        )
        instant = candidate.replace(tzinfo=tz, fold=0).astimezone(timezone.utc)
        if instant.astimezone(tz).replace(tzinfo=None) == candidate and instant > now:
            return instant
    raise ValueError("No daily schedule occurrence in the next five days")


async def execute_dream(
    config: dict,
    *,
    date: str = "",
    hint: str | None = None,
    origin: str = "manual",
    scheduled_at: str = "",
) -> str:
    """Serialize manual and scheduled runs across MCP processes in this Codex profile."""
    day = date or datetime.now(ZoneInfo(config["timezone"])).date().isoformat()
    if calendar_date.fromisoformat(day).isoformat() != day:
        raise ValueError("date must use YYYY-MM-DD")
    state = endpoint_dir(config)
    with lock(state / "dream.lock") as acquired:
        if not acquired:
            raise RuntimeError("Consolidation is already running for this endpoint in this Codex profile.")
        receipt = state / "dream_receipt.json"
        if scheduled_at:
            if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("scheduled_at") == scheduled_at:
                return "This scheduled run was already attempted."
            # Claim before the request: a lost acknowledgement must not trigger repeated daily runs.
            write_json(receipt, {"scheduled_at": scheduled_at})
        run = {"status": "running", "origin": origin, "date": day, "started_at": time.time()}
        write_json(state / "dream.json", run)
        log_status("dream_started", config=config)
        try:
            result = await call_async(
                config,
                "auto_dream",
                {"date": day, "hint": config["dreamHint"] if hint is None else hint},
                config["backgroundTimeoutMs"] / 1000,
            )
        except BaseException as exc:
            run.update(status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed")
            run["error"] = type(exc).__name__
            log_status("dream_" + run["status"], exc, config=config)
            raise
        else:
            run["status"] = "completed"
            log_status("dream_completed", config=config)
            answer = result["answer"]
            return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        finally:
            run["completed_at"] = time.time()
            write_json(state / "dream.json", run)


class ReMeRuntime:
    """Own one bounded task; elect a single scheduler among a profile's MCP processes."""

    def __init__(self):
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._changed = asyncio.Event()

    async def start(self) -> None:
        """Start only inside the FastMCP lifespan."""
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._serve())

    def settings_changed(self) -> None:
        """Wake this process immediately; other processes observe saved settings within a second."""
        self._changed.set()

    async def close(self) -> None:
        """Stop the timer and bound an in-flight dream by the configured shutdown budget."""
        self._stop.set()
        self._changed.set()
        if self._task is None:
            return
        try:
            timeout = load_config()["shutdownTimeoutMs"] / 1000
        except (OSError, ValueError):
            timeout = 5
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except TimeoutError:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        finally:
            self._task = None

    async def _wait(self, seconds: float = 1) -> None:
        try:
            await asyncio.wait_for(self._changed.wait(), max(0.01, seconds))
        except TimeoutError:
            pass
        self._changed.clear()

    async def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                with lock(data_dir() / "scheduler.lock") as acquired:
                    if acquired:
                        await self._lead()
            except (OSError, ValueError, RuntimeError) as exc:
                log_status("scheduler_failed", exc)
            if not self._stop.is_set():
                await self._wait()

    async def _lead(self) -> None:
        previous, due = None, None
        state = data_dir() / "scheduler.json"
        try:
            while not self._stop.is_set():
                config = load_config()
                key = tuple(config[name] for name in ("mcpUrl", "autoDreamEnabled", "dreamCron", "timezone"))
                if key != previous:
                    due = (
                        next_daily_run(config["dreamCron"], config["timezone"]) if config["autoDreamEnabled"] else None
                    )
                    previous = key
                    write_json(
                        state,
                        {
                            "phase": "running",
                            "endpoint": digest(config["endpoint"]),
                            "enabled": config["autoDreamEnabled"],
                            "cron": config["dreamCron"],
                            "timezone": config["timezone"],
                            "next_run_at": due.isoformat() if due else None,
                        },
                    )
                if due and datetime.now(timezone.utc) >= due:
                    write_json(
                        state,
                        {
                            "phase": "running",
                            "endpoint": digest(config["endpoint"]),
                            "enabled": config["autoDreamEnabled"],
                            "cron": config["dreamCron"],
                            "timezone": config["timezone"],
                            "next_run_at": None,
                        },
                    )
                    try:
                        await execute_dream(config, origin="scheduled", scheduled_at=due.isoformat())
                    except Exception as exc:
                        log_status("scheduled_dream_failed", exc, config=config)
                    previous = None
                    continue
                await self._wait(min(1, (due - datetime.now(timezone.utc)).total_seconds()) if due else 1)
        finally:
            write_json(state, {"phase": "stopped", "next_run_at": None})
