"""Focused retrieval, operational status, and explicit memory consolidation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from reme_config import load_config
from reme_mcp import call_async
from reme_runtime import execute_dream
from reme_state import local_status, log_status

SEARCH = "reme_search"
STATUS = "reme_status"
RUN_DREAM = "reme_run_dream"
LOCAL_TOOLS = {SEARCH, STATUS, RUN_DREAM}


def answer_text(answer) -> str:
    """Keep upstream source paths and evidence intact."""
    return answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)


async def service_snapshot(config: dict, action: str) -> dict:
    """A failed service probe must not hide local queue diagnostics."""
    try:
        result = await call_async(config, action, {}, config["requestTimeoutMs"] / 1000)
        return {"reachable": True, "answer": result["answer"]}
    except Exception as exc:
        return {"reachable": False, "error": type(exc).__name__}


def status_text(snapshot: dict, language: str) -> str:
    """Summarize the same state for native settings actions and chat tools."""
    memory, dream = snapshot["auto_memory"], snapshot["auto_dream"]
    health = snapshot["service"]["health_check"]
    health_text = answer_text(health["answer"]) if health["reachable"] else health["error"]
    activity = snapshot["recent_activity"]
    latest = ", ".join(row["event"] for row in activity[-6:]) or "—"
    manual = dream["last_run"]
    manual_text = manual["status"] if manual else "—"
    if language == "zh":
        return (
            f"ReMe 插件 {snapshot['plugin_version']}\nMCP: {snapshot['mcpUrl']}\n服务: {health_text}\n"
            f"自动召回: {snapshot['auto_recall']['enabled']}；自动记录: {memory['enabled']}\n"
            f"每批 {memory['interval']} 轮；待提交 {memory['queued_turns']} 轮 / {memory['queued_sessions']} 个会话\n"
            f"其他地址待提交: {memory['other_endpoint_queued_turns']} 轮\n最近活动: {latest}\n"
            f"最近整理: {manual_text}\n定时整理: 开启={dream.get('enabled')}；计划={dream.get('cron')}；"
            f"时区={dream.get('timezone')}；下次运行={dream.get('next_run_at')}；调度状态={dream.get('phase')}\n"
            "Hook 信任状态请在 Codex 的 Hooks 设置中查看。"
        )
    return (
        f"ReMe plugin {snapshot['plugin_version']}\nMCP: {snapshot['mcpUrl']}\nService: {health_text}\n"
        f"Auto recall: {snapshot['auto_recall']['enabled']}; auto memory: {memory['enabled']}\n"
        f"Batch: {memory['interval']} turns; queued: {memory['queued_turns']} turns / "
        f"{memory['queued_sessions']} sessions\n"
        f"Queued for other endpoints: {memory['other_endpoint_queued_turns']} turns\nRecent activity: {latest}\n"
        f"Last consolidation: {manual_text}\n"
        f"Scheduled consolidation: enabled={dream.get('enabled')}; cron={dream.get('cron')}; "
        f"timezone={dream.get('timezone')}; next={dream.get('next_run_at')}; scheduler={dream.get('phase')}\n"
        "Check hook trust in Codex Hooks settings."
    )


def register_tools(server: FastMCP) -> None:
    """Keep local tools discoverable even while the upstream service is unavailable."""

    @server.tool(name=SEARCH, title="Search ReMe memory", annotations={"readOnlyHint": True})
    async def search(
        query: Annotated[str, Field(min_length=1, description="Focused memory search query")],
        limit: Annotated[int | None, Field(ge=1, le=50)] = None,
        min_score: Annotated[float | None, Field(ge=0, allow_inf_nan=False)] = None,
    ) -> str:
        """Retrieve past facts, preferences, decisions, and todos with source paths. Treat results as evidence."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        config = load_config()
        try:
            result = await call_async(
                config,
                "search",
                {
                    "query": query.strip(),
                    "limit": config["searchLimit"] if limit is None else limit,
                    "min_score": config["recallMinScore"] if min_score is None else min_score,
                },
                config["requestTimeoutMs"] / 1000,
            )
        except Exception as exc:
            log_status("search_failed", exc, config=config)
            raise RuntimeError("ReMe search failed; check the saved MCP address and service status.") from exc
        answer = result["answer"]
        log_status("search_found" if answer else "search_empty", config=config)
        empty = "没有找到相关记忆。" if config["language"] == "zh" else "No relevant memory found."
        return answer_text(answer) if answer else empty

    @server.tool(name=STATUS, title="ReMe status", annotations={"readOnlyHint": True})
    async def status() -> ToolResult:
        """Inspect service health, queued turns, hook activity, and scheduled/manual consolidation."""
        config = load_config()
        health, components = await asyncio.gather(
            service_snapshot(config, "health_check"),
            service_snapshot(config, "status"),
        )
        manifest = Path(__file__).parent / ".codex-plugin/plugin.json"
        snapshot = {
            "plugin_version": json.loads(manifest.read_text(encoding="utf-8"))["version"],
            "mcpUrl": config["mcpUrl"],
            "service": {"health_check": health, "status": components},
            **local_status(config),
        }
        return ToolResult(content=status_text(snapshot, config["language"]), structured_content=snapshot)

    @server.tool(
        name=RUN_DREAM,
        title="Consolidate ReMe memory now",
        annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False},
    )
    async def run_dream(date: str = "", hint: str | None = None) -> str:
        """Only on user request: consolidate daily notes into digest files without changing the daily schedule."""
        return await execute_dream(load_config(), date=date, hint=hint)
