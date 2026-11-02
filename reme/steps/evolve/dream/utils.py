"""Shared auto-dream helpers."""

import datetime as dt
import re
from pathlib import Path

import yaml

from .._evolve import now
from ...base_step import BaseStep
from ....schema import DreamState


def state_from_context(step: BaseStep) -> DreamState:
    """Get dream state from context."""
    assert step.context is not None
    raw = step.context.get("dream") or step.context.response.metadata.get("dream") or {}
    state = DreamState.model_validate(raw)
    if not state.daily_dir:
        state.daily_dir = step.config_value("daily_dir")
    return state


def store_state(step: BaseStep, state: DreamState) -> None:
    """Store dream state in context."""
    assert step.context is not None
    data = state.model_dump()
    step.context["dream"] = data
    step.context.response.metadata["dream"] = data


def workspace_dir(step: BaseStep) -> Path:
    """Get workspace directory."""
    return step.file_store.workspace_path.resolve()


def daily_dir(step: BaseStep) -> str:
    """Get daily directory."""
    return step.config_value("daily_dir")


def today(step: BaseStep, explicit: str = "") -> str:
    """Get today's date."""
    if explicit.strip():
        return explicit.strip()
    tz = step.app_context.app_config.timezone if step.app_context is not None else None
    return now(tz).strftime("%Y-%m-%d")


def recent_dates(day: str, n_days: int) -> list[str]:
    """Return the inclusive recent-date window ending at ``day``."""
    try:
        base = dt.date.fromisoformat(day)
    except ValueError:
        return [day] if day else []
    n = max(int(n_days or 1), 1)
    return [(base - dt.timedelta(days=i)).isoformat() for i in range(n - 1, -1, -1)]


def llm_available(step: BaseStep) -> bool:
    """Check if LLM is available."""
    try:
        return step.as_llm is not None and step.agent_wrapper is not None
    except Exception:
        return False


def scan_day_files(workspace: Path, day: str, daily: str) -> list[str]:
    """Scan Markdown day-index and note files."""
    out: list[str] = []
    day_index = workspace / daily / f"{day}.md"
    if day_index.is_file():
        out.append(day_index.relative_to(workspace).as_posix())
    daily_root = workspace / daily / day
    if daily_root.is_dir():
        out.extend(p.relative_to(workspace).as_posix() for p in sorted(daily_root.rglob("*.md")) if p.is_file())
    return out


def pack_paths(
    workspace: Path,
    paths: list[str],
    *,
    limit_per_file: int = 60000,
    max_total_chars: int | None = None,
) -> str:
    """Pack paths into a single string.

    With ``max_total_chars`` set, blocks are packed in the given order until
    the accumulated size would exceed the budget; the first file is always
    kept and a trailer records how many files were omitted.
    """
    blocks: list[str] = []
    total = 0
    for index, rel in enumerate(paths):
        target = workspace / rel
        if not target.is_file():
            block = f"### {rel}\n(file not found)\n"
        else:
            try:
                text = target.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                block = f"### {rel}\n(error reading: {type(e).__name__}: {e})\n"
            else:
                suffix = "\n\n[truncated]\n" if len(text) > limit_per_file else ""
                block = f"### {rel}\n{text[:limit_per_file]}{suffix}\n"
        if max_total_chars is not None and index > 0 and total + len(block) > max_total_chars:
            omitted = len(paths) - index
            blocks.append(f"(omitted {omitted} file(s) to stay within the {max_total_chars}-char total budget)")
            break
        blocks.append(block)
        total += len(block)
    return "\n".join(blocks)


def _file_size(workspace: Path, rel: str) -> int:
    """Best-effort byte size of one workspace-relative path."""
    try:
        return (workspace / rel).stat().st_size
    except OSError:
        return 0


def _day_key(rel: str) -> str:
    """Group key that keeps one scanned day together (``daily/<date>``)."""
    parts = rel.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else rel


def batch_paths(
    workspace: Path,
    paths: list[str],
    *,
    max_chars: int,
) -> list[list[str]]:
    """Split paths into size-capped batches without dropping any path.

    Files belonging to the same scanned day are kept in one batch whenever the
    day fits the budget, so cross-file extraction still sees a whole day. A day
    larger than the budget is split by file size, and a single oversized file
    keeps a batch of its own (``pack_paths`` still applies its per-file cap).

    Every path lands in exactly one batch; callers must never mark an unseen
    path as extracted, which is why this returns a partition and not a slice.
    """
    if not paths:
        return []
    if max_chars <= 0:
        return [list(paths)]

    groups: list[list[str]] = []
    for rel in paths:
        if groups and _day_key(groups[-1][0]) == _day_key(rel):
            groups[-1].append(rel)
        else:
            groups.append([rel])

    batches: list[list[str]] = []
    current: list[str] = []
    total = 0
    for group in groups:
        group_size = sum(_file_size(workspace, rel) for rel in group)
        if current and total + group_size > max_chars:
            batches.append(current)
            current, total = [], 0
        if group_size <= max_chars:
            current.extend(group)
            total += group_size
            continue
        for rel in group:
            size = _file_size(workspace, rel)
            if current and total + size > max_chars:
                batches.append(current)
                current, total = [], 0
            current.append(rel)
            total += size
    if current:
        batches.append(current)
    return batches


def clean_paths(raw_paths, allowed: set[str]) -> list[str]:
    """Clean paths."""
    if not isinstance(raw_paths, list):
        return []
    out: list[str] = []
    for item in raw_paths:
        path = str(item or "").strip()
        if path in allowed and path not in out:
            out.append(path)
    return out


def previous_dates(day: str, n_days: int) -> list[str]:
    """Get previous dates."""
    try:
        base = dt.date.fromisoformat(day)
    except ValueError:
        return []
    return [(base - dt.timedelta(days=i)).isoformat() for i in range(1, max(n_days, 0) + 1)]


def parse_structured_reply(text: str, *, strict: bool = False) -> dict:
    """Parse a JSON/YAML object; strict mode rejects lossy fallbacks with diagnostics."""
    candidates = [text.strip()]
    candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json|ya?ml)?\s*(.*?)```", text, re.S | re.I))
    error = "expected a JSON/YAML object"
    for raw in candidates:
        if not raw:
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            error = f"{type(exc).__name__}: {exc}"
            if strict:
                continue
            data = _parse_scalar_mapping(raw)
        if isinstance(data, dict) and (data or strict):
            return data
    if strict:
        raise ValueError(f"Invalid structured reply: {error}")
    return {}


def _parse_scalar_mapping(raw: str) -> dict:
    """Parse a scalar mapping."""
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if match := re.match(r"^\s*(action|target_path|note)\s*:\s*(.+?)\s*$", line):
            out[match.group(1)] = match.group(2).strip().strip("\"'")
    return out
