"""File, state, and formatting helpers for Auto Fin."""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiofiles
import frontmatter

from ....schema import Checkpoint, PortfolioSnapshot
from ...file_io._file_io import get_path_lock

_CHECKPOINT_LABELS = {
    Checkpoint.OPEN: "09:00",
    Checkpoint.MIDDAY: "11:45",
    Checkpoint.CLOSE: "14:45",
}
_SECTION_RE = re.compile(r"(?ms)^## (09:00|11:45|14:45)\s*\n.*?(?=^## (?:09:00|11:45|14:45)\s*$|\Z)")


async def write_atomic(path: Path, content: str) -> None:
    """Write one UTF-8 file via sibling temp and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = await get_path_lock(path)
    async with lock:
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as stream:
                await stream.write(content)
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _lock_owner_is_alive(path: Path) -> bool:
    """Return whether a lock file names a process that still exists."""
    try:
        owner = json.loads(path.read_text(encoding="utf-8"))
        pid = int(owner["pid"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@asynccontextmanager
async def checkpoint_lock(path: Path, run_id: str):
    """Hold an in-process and recoverable cross-process checkpoint lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    in_process = await get_path_lock(path)
    async with in_process:
        payload = json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "started_at": datetime.now().astimezone().isoformat(),
            },
            ensure_ascii=False,
        ).encode()
        descriptor = -1
        for attempt in range(2):
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError as exc:
                if attempt == 0 and not _lock_owner_is_alive(path):
                    path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(f"Auto Fin checkpoint is already locked: {run_id}") from exc
        if descriptor < 0:  # pragma: no cover - the loop either acquires or raises.
            raise RuntimeError(f"failed to acquire Auto Fin checkpoint lock: {run_id}")
        try:
            os.write(descriptor, payload)
            os.close(descriptor)
            descriptor = -1
            yield
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            path.unlink(missing_ok=True)


def load_document(path: Path) -> frontmatter.Post:
    """Load a Markdown document or return an empty post."""
    if not path.is_file():
        return frontmatter.Post("")
    try:
        return frontmatter.load(path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid Auto Fin Markdown document: {path}") from exc


def _replace_section(body: str, checkpoint: Checkpoint, section: str, title: str) -> str:
    label = _CHECKPOINT_LABELS[checkpoint]
    replacement = f"## {label}\n\n{section.strip()}\n"
    matches = list(_SECTION_RE.finditer(body))
    sections = {match.group(1): match.group(0).strip() for match in matches}
    sections[label] = replacement.strip()
    order = ("09:00", "11:45", "14:45")
    rendered = "\n\n".join(sections[key] for key in order if key in sections)
    return f"# {title}\n\n{rendered}\n"


async def upsert_report(
    path: Path,
    *,
    document_type: str,
    trade_date: str,
    timezone: str,
    checkpoint: Checkpoint,
    run: dict[str, Any],
    section: str,
    title: str,
) -> None:
    """Replace one run and its matching checkpoint section."""
    document = load_document(path)
    runs = [value for value in document.metadata.get("runs", []) if value.get("run_id") != run["run_id"]]
    runs.append(run)
    runs.sort(key=lambda value: str(value.get("decision_at", "")))
    metadata = {
        "schema_version": "auto-fin/v1",
        "document_type": document_type,
        "trade_date": trade_date,
        "timezone": timezone,
        "updated_at": run["generated_at"],
        "runs": runs,
    }
    body = _replace_section(document.content, checkpoint, section, title)
    rendered = frontmatter.dumps(frontmatter.Post(body.rstrip(), **metadata))
    await write_atomic(path, f"{rendered.rstrip()}\n")


def find_run(path: Path, run_id: str) -> dict[str, Any] | None:
    """Return one persisted run by id."""
    for run in load_document(path).metadata.get("runs", []):
        if run.get("run_id") == run_id:
            return run
    return None


def latest_portfolio_snapshot(
    workspace: Path,
    daily_dir: str,
    *,
    before: datetime,
    excluding_run_id: str,
) -> PortfolioSnapshot | None:
    """Rebuild the latest valid snapshot strictly before the current decision."""
    candidates: list[tuple[datetime, dict[str, Any]]] = []
    root = workspace / daily_dir
    if not root.is_dir():
        return None
    for path in root.glob("*/portfolio.md"):
        for run in load_document(path).metadata.get("runs", []):
            if run.get("run_id") == excluding_run_id or not isinstance(run.get("snapshot"), dict):
                continue
            try:
                decision_at = datetime.fromisoformat(str(run["decision_at"]))
            except (KeyError, TypeError, ValueError):
                continue
            if decision_at < before:
                candidates.append((decision_at, run))
    if not candidates:
        return None
    _, latest = max(candidates, key=lambda value: value[0])
    return PortfolioSnapshot.model_validate(latest["snapshot"])


def analysis_section(description: str, body: str, limitations: list[str]) -> str:
    """Render one readable analysis checkpoint section."""
    chunks = [description.strip(), body.strip()]
    if limitations:
        chunks.append("### 限制与失效条件\n\n" + "\n".join(f"- {value}" for value in limitations))
    return "\n\n".join(value for value in chunks if value)


def portfolio_section(
    snapshot_before: PortfolioSnapshot,
    snapshot_after: PortfolioSnapshot,
    settlements: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    agent_body: str,
    *,
    interval_return: float,
    status: str,
    us_as_of: str,
) -> str:
    """Render deterministic tables plus the analysis rationale."""
    lines = [
        f"**运行状态：{status}**",
        "",
        "> 纯模拟盘，不会执行真实交易，也不构成投资建议。",
        "",
        "| 指标 | 结算前 | 当前 |",
        "|---|---:|---:|",
        f"| 组合净值 | {snapshot_before.nav:.6f} | {snapshot_after.nav:.6f} |",
        f"| 现金净值 | {snapshot_before.cash_nav:.6f} | {snapshot_after.cash_nav:.6f} |",
        f"| 持仓数 | {len(snapshot_before.positions)} | {len(snapshot_after.positions)} |",
        f"| 本区间组合收益 | - | {interval_return:.4%} |",
        "",
        "### 当前持仓",
        "",
        "| 标的 | 类型 | 本区间 | 累计 | 归一化价值 | 组合贡献 | 可卖日期 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    if snapshot_after.positions:
        for position in snapshot_after.positions:
            lines.append(
                f"| {position.name} ({position.code}) | {position.instrument_type} | "
                f"{position.interval_return:.4%} | {position.cumulative_return:.4%} | "
                f"{position.normalized_value:.6f} | {position.portfolio_contribution:.6f} | "
                f"{position.eligible_sell_date.isoformat()} |",
            )
    else:
        lines.append("| 无 | - | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "### 已结算操作",
            "",
            "| 操作 | 标的 | 状态 | 成交基准 | 原因 |",
            "|---|---|---|---|---|",
        ],
    )
    for item in settlements:
        lines.append(
            f"| {item['action']} | {item['code']} | {item['status']} | "
            f"{item['fill_basis']} | {item.get('reason') or '-'} |",
        )
    if not settlements:
        lines.append("| - | - | 无待结算操作 | - | - |")

    lines.extend(
        [
            "",
            "### 新操作建议",
            "",
            "| 操作 | 标的 | 状态 | 计划成交 | 置信度 | 理由 |",
            "|---|---|---|---|---:|---|",
        ],
    )
    for item in accepted:
        lines.append(
            f"| {item['action']} | {item['code']} | {item['status']} | "
            f"{item.get('scheduled_fill_at') or '-'} | {item['confidence']:.2f} | {item['reason']} |",
        )
    if not accepted:
        lines.append("| HOLD | - | FILLED | - | - | 无新的合法操作 |")

    lines.extend(
        [
            "",
            "### 被拒绝操作",
            "",
            "| 操作 | 标的 | 状态 | 原因 |",
            "|---|---|---|---|",
        ],
    )
    for item in rejected:
        lines.append(
            f"| {item['action']} | {item['code']} | {item['status']} | {item['rejection_reason']} |",
        )
    if not rejected:
        lines.append("| - | - | - | 无 |")

    lines.extend(
        [
            "",
            "### 分析摘要",
            "",
            agent_body.strip(),
            "",
            f"美股关联分析复用时间：{us_as_of or '不可用'}。",
        ],
    )
    return "\n".join(lines)
