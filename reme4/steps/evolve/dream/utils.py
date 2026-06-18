"""Shared auto-dream helpers."""

import datetime as dt
import re
from pathlib import Path

import yaml

from .._evolve import now
from ...base_step import BaseStep
from .schema import DreamState


def state_from_context(step: BaseStep) -> DreamState:
    assert step.context is not None
    raw = step.context.get("dream") or step.context.response.metadata.get("dream") or {}
    return DreamState.model_validate(raw)


def store_state(step: BaseStep, state: DreamState) -> None:
    assert step.context is not None
    data = state.model_dump()
    step.context["dream"] = data
    step.context.response.metadata["dream"] = data


def vault_dir(step: BaseStep) -> Path:
    vr = getattr(step.file_store, "vault_path", None)
    return Path(vr).resolve() if vr else Path.cwd().resolve()


def daily_dir(step: BaseStep) -> str:
    cfg = step.app_context.app_config if step.app_context is not None else None
    return (cfg.daily_dir if cfg else "") or "daily"


def today(step: BaseStep, explicit: str = "") -> str:
    if explicit.strip():
        return explicit.strip()
    tz = step.app_context.app_config.timezone if step.app_context is not None else None
    return now(tz).strftime("%Y-%m-%d")


def llm_available(step: BaseStep) -> bool:
    try:
        return step.as_llm is not None and step.agent_wrapper is not None
    except Exception:
        return False


def scan_day_files(vault: Path, day: str, daily: str, interests_name: str = "interests.yaml") -> list[str]:
    out: list[str] = []
    day_index = vault / daily / f"{day}.md"
    if day_index.is_file():
        out.append(str(day_index.relative_to(vault)))
    daily_root = vault / daily / day
    if daily_root.is_dir():
        out.extend(str(p.relative_to(vault)) for p in sorted(daily_root.rglob("*.md")) if p.is_file())
    return [p for p in out if p != f"{daily}/{day}/{interests_name}"]


def pack_paths(vault: Path, paths: list[str], *, limit_per_file: int = 60000) -> str:
    blocks: list[str] = []
    for rel in paths:
        target = vault / rel
        if not target.is_file():
            blocks.append(f"### {rel}\n(file not found)\n")
            continue
        try:
            text = target.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            blocks.append(f"### {rel}\n(error reading: {type(e).__name__}: {e})\n")
            continue
        suffix = "\n\n[truncated]\n" if len(text) > limit_per_file else ""
        blocks.append(f"### {rel}\n{text[:limit_per_file]}{suffix}\n")
    return "\n".join(blocks)


def clean_paths(raw_paths, allowed: set[str]) -> list[str]:
    if not isinstance(raw_paths, list):
        return []
    out: list[str] = []
    for item in raw_paths:
        path = str(item or "").strip()
        if path in allowed and path not in out:
            out.append(path)
    return out


def normalize_topic(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text.lower()).strip()


def previous_dates(day: str, n_days: int) -> list[str]:
    try:
        base = dt.date.fromisoformat(day)
    except ValueError:
        return []
    return [(base - dt.timedelta(days=i)).isoformat() for i in range(1, max(n_days, 0) + 1)]


def load_yaml_topics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        return []
    return [cleaned for t in topics if isinstance(t, dict) and (cleaned := clean_topic(t))]


def clean_topic(raw: dict) -> dict:
    title, reason = str(raw.get("title") or "").strip(), str(raw.get("reason") or "").strip()
    if not title or not reason:
        return {}
    keywords = raw.get("keywords") or []
    paths = raw.get("paths") or []
    return {
        "title": title,
        "reason": reason,
        "evidence": str(raw.get("evidence") or "").strip(),
        "keywords": [str(k).strip() for k in keywords if str(k).strip()] if isinstance(keywords, list) else [],
        "paths": [str(p).strip() for p in paths if str(p).strip()] if isinstance(paths, list) else [],
    }


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    path.write_text(rendered if rendered.endswith("\n") else f"{rendered}\n", encoding="utf-8")


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
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if match := re.match(r"^\s*(action|target_path|note)\s*:\s*(.+?)\s*$", line):
            out[match.group(1)] = match.group(2).strip().strip("\"'")
    return out
