"""Shared auto-dream helpers."""

import datetime as dt
import json
import re
from pathlib import Path

import yaml

from .._evolve import now, strip_frontmatter
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

    YAML front matter is stripped from each file; only the body is packed.

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
                raw = target.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                block = f"### {rel}\n(error reading: {type(e).__name__}: {e})\n"
            else:
                # Daily notes carry YAML front matter (``name``/``description``/``tags``).
                # Packing it verbatim makes the extraction model mimic that shape and emit
                # units without ``bucket``/``summary``, which then fail DreamUnit validation
                # (issue #730). Only the body is relevant material.
                text = strip_frontmatter(raw)[1]
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
    # Parse complete documents before envelope recovery so YAML flow mappings and
    # literal reasoning tags in string values retain their original structure.
    candidates = [text, _strip_reasoning_tags(text)]
    error = "expected a JSON/YAML object"
    for raw in candidates:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                if strict:
                    raise ValueError("Invalid structured reply: expected a JSON/YAML object")
                return {}

    cleaned = candidates[-1]
    # Split at fences, retaining both fenced and unfenced regions in response
    # order. A final bare answer must outrank an earlier fenced draft, too.
    candidates = _reply_regions(cleaned)
    candidates.reverse()
    for raw in candidates:
        if not raw.strip():
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            error = f"{type(exc).__name__}: {exc}"
            data = None
        # Complete YAML roots take precedence over JSON nested within that
        # same region, never over a later region of the response.
        if not isinstance(data, (dict, list)):
            data = _final_json_value(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            if strict:
                raise ValueError("Invalid structured reply: expected a JSON/YAML object")
            return {}
    if not strict:
        for raw in candidates:
            data = _parse_scalar_mapping(raw)
            if data:
                break
        return data
    raise ValueError(f"Invalid structured reply: {error}")


def _strip_reasoning_tags(raw: str) -> str:
    """Drop paired reasoning contents; preserve quoted payloads and orphan answers."""
    # Fences and JSON strings are opaque to the envelope scanner. Attributes
    # may themselves be quoted; a fence inside a reasoning block is still removed.
    tokens = re.finditer(r'```.*?```|"(?:\\.|[^"\\])*"|<\s*(/?)\s*(think|thought)\b[^>]*>', raw, re.I | re.S)
    removed: list[tuple[int, int]] = []
    opened: list[tuple[str, int, int]] = []
    for token in tokens:
        if token.group(1) is None:
            continue
        if not token.group(1):
            opened.append((token.group(2).lower(), token.start(), token.end()))
        elif opened and opened[-1][0] == token.group(2).lower():
            _, start, _ = opened.pop()
            removed.append((start, token.end()))
        else:
            # A lone closing marker separates a reasoning prefix from its answer.
            removed.append((0, token.end()))
        if token.group(1) and not opened:
            # Once an envelope ends, preserve a complete YAML/JSON answer,
            # including unquoted or block-scalar examples of reasoning tags.
            try:
                answer = yaml.safe_load(raw[token.end() :])
            except yaml.YAMLError:
                pass
            else:
                if isinstance(answer, (dict, list)):
                    break
    # Some providers omit the closing marker but still append the final answer.
    removed.extend((start, end) for _, start, end in opened)
    result = []
    cursor = 0
    for start, end in sorted(removed):
        if start >= cursor:
            result.append(raw[cursor:start])
        cursor = max(cursor, end)
    result.append(raw[cursor:])
    return "".join(result)


def _reply_regions(raw: str) -> list[str]:
    """Split Markdown fences without treating JSON string contents as fences."""
    fence = re.compile(r"```(?:json|ya?ml)?\s*(.*?)```", re.S | re.I)
    decoder = json.JSONDecoder()
    regions = []
    start = index = 0
    while index < len(raw):
        if raw[index] == '"':
            try:
                _, index = decoder.raw_decode(raw, index)
                continue
            except json.JSONDecodeError:
                pass
        if match := fence.match(raw, index):
            regions.extend((raw[start:index], match.group(1)))
            start = index = match.end()
        else:
            index += 1
    regions.append(raw[start:])
    return regions


def _final_json_value(raw: str) -> dict | list | None:
    """Recover the final independent JSON container, preserving valid outer values."""
    decoder = json.JSONDecoder()
    final = None
    index = 0
    while index < len(raw):
        if raw[index] not in '{["':
            index += 1
            continue
        try:
            data, end = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            # A malformed draft must not carry its string/brace state into an
            # independently valid final answer. Retry from the next character.
            index += 1
            continue
        if isinstance(data, (dict, list)):
            final = data
        # Valid containers and strings are opaque: do not promote their nested
        # objects or JSON-looking string contents into independent answers.
        index = end
    return final


def _parse_scalar_mapping(raw: str) -> dict:
    """Parse a scalar mapping."""
    out: dict[str, str] = {}
    for line in raw.splitlines():
        if match := re.match(r"^\s*(action|target_path|note)\s*:\s*(.+?)\s*$", line):
            out[match.group(1)] = match.group(2).strip().strip("\"'")
    return out
