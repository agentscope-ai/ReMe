"""Run current chat and seven-query proactive jobs with controlled clock and evidence."""

# pylint: disable=protected-access

import asyncio
import json
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from reme.application import Application
from reme.schema import SessionMessage, SessionSnapshot, SessionStatus, collect_reply
from reme.steps.base_step import BaseStep
from reme.steps.proactive import ProactiveAgentStep, CronAgentStep

from benchmark.proactive.cases import SESSION_ID, frozen_now, seed

EFFORT = "xhigh"


def plain(value):
    """Convert schema objects to JSON-compatible values."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=lambda obj: obj.model_dump(mode="json")))


def save(path: Path, value) -> None:
    """Atomically replace one recorded benchmark result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def files(root: Path) -> dict[str, str]:
    """Snapshot visible text fixtures without following symlinks."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file() or relative.parts[0].startswith("."):
            continue
        try:
            result[relative.as_posix()] = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
    return result


class FrozenProactive(ProactiveAgentStep):
    """Use the fixture clock for proactive queries."""

    def _now(self):
        return frozen_now()


class FrozenCron(CronAgentStep):
    """Use the fixture clock for task queries."""

    def _now(self):
        return frozen_now()


class FixtureConnector(BaseStep):
    """Return synthetic evidence without accessing external data."""

    async def execute(self):
        """Run one case and retain its outcomes and workspace evidence."""
        self.context.response.answer = self.kwargs["payload"]
        return self.context.response


def configuration(base: dict, model: str, workspace: Path, bundle: dict, case: dict) -> dict:
    """Adapt production jobs to one isolated benchmark case."""
    cfg = deepcopy(base)
    names = (
        "pa_chat",
        "pa_session",
        "pa_new_session",
        "pa_proactive",
        "cron_agent",
        "screen_capture",
        "finance_news",
        "read_conversation",
        "read_proactive_inbox",
        "get_chat_messages",
        "read_cron_results",
        "send_chat_message",
        "write_proactive_inbox",
        "manage_temp_proactive_agent",
        "check_json",
    )
    cfg["jobs"] = {name: cfg["jobs"][name] for name in names}
    cfg["components"] = {name: cfg["components"][name] for name in ("codex_app_server", "agent_wrapper", "pa")}
    cfg.update(
        workspace_dir=str(workspace),
        metadata_dir="reme_metadata",
        language="zh",
        service={"backend": "cli"},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
    )
    server = cfg["components"]["codex_app_server"]["default"]
    server["codex_home"] = ".codex"
    server["config_overrides"] = [
        item
        for item in server.get("config_overrides", [])
        if not any(word in item for word in ("memories", "subagent", "review_model", "web_search"))
    ] + [
        "features.memories=false",
        "memories.generate_memories=false",
        "memories.use_memories=false",
        "features.multi_agent=false",
        'web_search="disabled"',
    ]
    catalog = deepcopy(server["model_catalog"]["models"][0])
    catalog.update(slug=model, display_name=model, input_modalities=["text"])
    server["model_catalog"] = {"models": [catalog]}
    cfg["components"]["agent_wrapper"]["default"].update(model=model, effort=EFFORT)
    for name in ("pa_chat", "pa_new_session"):
        cfg["jobs"][name]["steps"][0]["prompt_dict"] = {**bundle["memory"], **bundle["pa"]}
    cfg["jobs"]["pa_proactive"]["steps"][0].update(
        backend="benchmark_proactive",
        prompt_dict={**bundle["memory"], **bundle["proactive"]},
    )
    cfg["jobs"]["cron_agent"]["steps"][0].update(
        backend="benchmark_cron",
        prompt_dict={**bundle["memory"], **bundle["cron"]},
    )
    for name in ("screen_capture", "finance_news"):
        cfg["jobs"][name]["steps"] = [
            {
                "backend": "benchmark_connector",
                "payload": {
                    "context": "Synthetic benchmark evidence; read paths for complete contents.",
                    "paths": list(case.get("resources", {})),
                },
            },
        ]
    return cfg


def bind_owner(agent, case: dict, record: dict) -> None:
    """Replace fixture-session I/O while preserving other native sessions."""
    snapshot = SessionSnapshot(
        session_id=SESSION_ID,
        state=SessionStatus(session_id=SESSION_ID, status=case.get("session_status", "idle")),
        messages=[SessionMessage.model_validate(row) for row in case.get("messages", [])],
        last_turn_id="fixture-turn",
    )
    read, status, append, reply = agent.read_session, agent.get_session_status, agent.append_session, agent.reply

    async def reply_with_fixture(inputs, **kwargs):
        if kwargs.get("fork_session") and kwargs.get("resume") == SESSION_ID:
            # The synthetic owner is not a native thread. This harness exercises summary preparation,
            # but native fork behavior belongs to provider integration validation.
            kwargs.pop("resume")
            kwargs.pop("fork_session")
            dialogue = "\n".join(f"[{row.role}] {row.content}" for row in snapshot.messages if row.source == "chat")
            inputs = f"Synthetic source conversation:\n{dialogue}\n\n{inputs}"
            record["synthetic_context_preparation"] = True
        return await reply(inputs, **kwargs)

    async def read_session(session_id):
        return snapshot.model_copy(deep=True) if session_id == SESSION_ID else await read(session_id)

    async def get_status(session_id):
        return snapshot.state if session_id == SESSION_ID else await status(session_id)

    async def append_session(session_id, content, *, message_id, **kwargs):
        if session_id != SESSION_ID:
            return await append(session_id, content, message_id=message_id, **kwargs)
        existing = next((row for row in snapshot.messages if row.id == message_id), None)
        if existing:
            return existing
        if snapshot.state.status != "idle":
            raise RuntimeError("Fixture chat is busy")
        message = SessionMessage(
            id=message_id,
            role="assistant",
            source="proactive",
            content=content,
            delivery=kwargs.get("delivery", "idle"),
            origin_session_id=session_id,
        )
        snapshot.messages.append(message)
        record["delivered"].append(message.model_dump(mode="json"))
        return message

    agent.read_session, agent.get_session_status, agent.append_session = read_session, get_status, append_session
    agent.reply = reply_with_fixture


def instrument(app, folder: Path, record: dict) -> None:
    """Record job execution and model events at their real boundaries."""
    agent = app.context.components["agent_wrapper"]["default"]
    original = agent.reply_stream

    async def stream(inputs, **kwargs):
        turn, started = {"input": plain(inputs)}, time.monotonic()
        record["turns"].append(turn)
        try:
            async for chunk in original(inputs, **kwargs):
                row = plain(chunk)
                with (folder / "trace.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if row["chunk_type"] in {"tool_call", "tool_result"}:
                    record["tool_events"].append(row)
                native = row.get("metadata", {}).get("native", {})
                if native.get("method") == "thread/tokenUsage/updated":
                    turn["native_token_usage"] = native.get("payload", {})
                if row["chunk_type"] == "reply_end":
                    turn.update(
                        {key: value for key, value in (row["chunk"] or {}).items() if key not in {"raw", "events"}},
                    )
                yield chunk
        finally:
            turn["duration_s"] = round(time.monotonic() - started, 3)

    agent.reply_stream = stream
    for name, job in app.context.jobs.items():
        original_steps = job._run_steps

        async def traced(context, job_name=name, run_steps=original_steps):
            event = {"name": job_name, "args": plain(context.data)}
            record["jobs"].append(event)
            try:
                response = await run_steps(context)
                event["response"] = plain(response)
                return response
            except BaseException as exc:
                event["error"] = str(exc) or type(exc).__name__
                raise

        job._run_steps = traced


async def execute(base: dict, model: str, bundle: dict, case: dict, folder: Path, timeout: float) -> dict:
    """Run one case and retain its outcomes and workspace evidence."""
    workspace = folder / "workspace"
    workspace.mkdir(parents=True)
    seed(workspace, case)
    record = {"fixture": case, "turns": [], "jobs": [], "tool_events": [], "delivered": [], "before": files(workspace)}
    cfg = configuration(base, model, workspace, bundle, case)
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            app = Application(**cfg)
            for name, cls in (
                ("benchmark_proactive", FrozenProactive),
                ("benchmark_cron", FrozenCron),
                ("benchmark_connector", FixtureConnector),
            ):
                app.context.registry.add(name, cls, owner="benchmark.proactive")
            runtime = app.context.components["pa"]["default"]
            runtime._now = frozen_now
            agent = app.context.components["agent_wrapper"]["default"]
            if case["role"] != "pa":
                bind_owner(agent, case, record)
            instrument(app, folder, record)
            async with app:
                if case["role"] == "pa":
                    for query in case["queries"]:
                        reply = await collect_reply(app.run_stream_job("pa_chat", query=query))
                        if reply["status"] != "completed":
                            raise RuntimeError(reply.get("error") or reply["status"])
                else:
                    # Simulate cron triggers before proactive wakes, using the real three-query task Job.
                    for task in await asyncio.to_thread(runtime.tasks.load):
                        if task.enabled:
                            response = await app.run_job("cron_agent", task=task.model_dump(mode="json"))
                            if not response.success:
                                raise RuntimeError(str(response.answer))
                    response = await app.run_job("pa_proactive")
                    record["response"] = plain(response)
                    if not response.success:
                        raise RuntimeError(str(response.answer))
                record["reminders"] = await runtime.get_chat_messages()
                record["completed"] = True
    except Exception as exc:
        record.update(completed=False, error=f"{type(exc).__name__}: {exc}")
    record.update(after=files(workspace), duration_s=round(time.monotonic() - started, 3))
    secret = cfg["components"]["codex_app_server"]["default"].get("api_key")
    return json.loads(json.dumps(record, ensure_ascii=False).replace(secret, "<redacted>")) if secret else record
