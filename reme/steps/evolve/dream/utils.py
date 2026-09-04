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
    vr = getattr(step.file_store, "workspace_path", None)
    return Path(vr).resolve() if vr else Path.cwd().resolve()


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


def scan_day_files(workspace: Path, day: str, daily: str, interests_name: str = "interests.yaml") -> list[str]:
    """Scan day files."""
    out: list[str] = []
    day_index = workspace / daily / f"{day}.md"
    if day_index.is_file():
        out.append(day_index.relative_to(workspace).as_posix())
    daily_root = workspace / daily / day
    if daily_root.is_dir():
        out.extend(p.relative_to(workspace).as_posix() for p in sorted(daily_root.rglob("*.md")) if p.is_file())
    return [p for p in out if p != f"{daily}/{day}/{interests_name}"]


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


def load_yaml_topics(path: Path, *, strict: bool = False) -> list[dict]:
    """Load YAML topics."""
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: {exc}") from exc
        return []
    if data is None:
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: expected an object")
        return []
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        if strict:
            raise ValueError(f"Invalid interests YAML at {path}: topics must be a list")
        return []
    cleaned_topics = []
    for index, topic in enumerate(topics):
        if strict:
            _validate_topic(topic, path, index)
        if isinstance(topic, dict) and (cleaned := clean_topic(topic)):
            cleaned_topics.append(cleaned)
    return cleaned_topics


def _validate_topic(topic: object, path: Path, index: int) -> None:
    """Reject topic data that would otherwise be silently discarded or coerced."""
    prefix = f"Invalid interests YAML at {path}: topics[{index}]"
    if not isinstance(topic, dict):
        raise ValueError(f"{prefix} must be an object")

    allowed = {"title", "reason", "evidence", "keywords", "paths"}
    if unknown := sorted(set(topic) - allowed):
        raise ValueError(f"{prefix} has unknown field(s): {', '.join(str(key) for key in unknown)}")

    for field in ("title", "reason"):
        value = topic.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}.{field} must be a non-empty string")
    if "evidence" in topic and not isinstance(topic["evidence"], str):
        raise ValueError(f"{prefix}.evidence must be a string")
    for field in ("keywords", "paths"):
        if field not in topic:
            continue
        values = topic[field]
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{prefix}.{field} must be a list of non-empty strings")


def clean_topic(raw: dict) -> dict:
    """Clean topic."""
    title, reason = (
        str(raw.get("title") or "").strip(),
        str(raw.get("reason") or "").strip(),
    )
    if not title or not reason:
        return {}
    keywords = raw.get("keywords") or []
    paths = raw.get("paths") or []
    return {
        "title": title,
        "reason": reason,
        "evidence": str(raw.get("evidence") or "").strip(),
        "keywords": ([str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else []),
        "paths": ([str(p).strip() for p in paths if str(p).strip()] if isinstance(paths, list) else []),
    }


def parse_structured_reply(text: str) -> dict:
    """Parse a JSON/YAML object from an agent reply, including fenced blocks."""
    candidates = [text.strip()]
    candidates.extend(m.group(1).strip() for m in re.finditer(r"```(?:json|ya?ml)?\s*(.*?)```", text, re.S | re.I))
    for raw in candidates:
        if not raw:
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            data = _parse_scalar_mapping(raw)
        if isinstance(data, dict) and data:
            return data
    return {}


def _parse_scalar_mapping(raw: str) -> dict:
    """Parse a scalar mapping."""
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if match := re.match(r"^\s*(action|target_path|note)\s*:\s*(.+?)\s*$", line):
            out[match.group(1)] = match.group(2).strip().strip("\"'")
    return out
