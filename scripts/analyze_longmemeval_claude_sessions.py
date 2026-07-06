#!/usr/bin/env python3
"""Analyze Claude Code tool usage for LongMemEval session-answer runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("/Users/yuli/workspace/ReMe/longmemeval")
DEFAULT_SESSION_NAME = "claude_code_session.jsonl"
DEFAULT_JUDGE_NAME = "longmemeval_session_answer_judge.json"
DEFAULT_JSON_OUTPUT = "claude_code_session_analysis.json"
DEFAULT_MD_OUTPUT = "claude_code_session_analysis.md"

SESSION_FILE_RE = re.compile(r"longmemeval/\d+/session/[^\s\"']+?\.json")
SESSION_BASENAME_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}_\([^)]*\)_[^\s\"']+?\.json\b")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def iter_index_dirs(dataset_dir: Path) -> list[Path]:
    return sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def parse_indices(value: str) -> set[int]:
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid index range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "thinking"):
                    value = item.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
        return "\n".join(chunks)
    return ""


def iter_message_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [item for item in content if isinstance(item, dict)]


def command_family(command: str) -> str:
    command = command.strip()
    if not command:
        return "empty"
    if re.search(r"\b(rg|grep)\b", command):
        return "search"
    if re.search(r"\bls\b", command):
        return "list"
    if re.search(r"\bpython3?\b", command):
        return "python"
    if re.search(r"\b(find|fd)\b", command):
        return "find"
    if re.search(r"\b(cat|sed|awk|jq|head|tail)\b", command):
        return "inspect"
    if re.search(r"\bwc\b", command):
        return "count"
    return command.split()[0]


def extract_session_files(value: str) -> set[str]:
    files: set[str] = set()
    for match in SESSION_FILE_RE.finditer(value):
        files.add(Path(match.group(0)).name)
    for match in SESSION_BASENAME_RE.finditer(value):
        files.add(match.group(0))
    return files


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1 - frac) + ordered[upper] * frac


def summarize_numeric(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def parse_session(path: Path) -> dict[str, Any]:
    tool_uses: dict[str, dict[str, Any]] = {}
    tools: list[dict[str, Any]] = []
    assistant_texts: list[str] = []
    thinking_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            message = payload.get("message")
            if isinstance(message, dict):
                usage = message.get("usage")
                if isinstance(usage, dict):
                    total_input_tokens += int(usage.get("input_tokens") or 0)
                    total_input_tokens += int(usage.get("cache_read_input_tokens") or 0)
                    total_input_tokens += int(usage.get("cache_creation_input_tokens") or 0)
                    total_output_tokens += int(usage.get("output_tokens") or 0)

                role = message.get("role")
                if role == "assistant":
                    text = text_from_content(message.get("content"))
                    if text:
                        assistant_texts.append(text)

                for item in iter_message_items(message):
                    item_type = item.get("type")
                    if item_type == "thinking":
                        thinking_count += 1
                    if item_type == "tool_use":
                        tool = {
                            "id": item.get("id"),
                            "name": item.get("name"),
                            "input": item.get("input") if isinstance(item.get("input"), dict) else {},
                            "result": "",
                        }
                        if isinstance(tool["id"], str):
                            tool_uses[tool["id"]] = tool
                        tools.append(tool)
                    elif item_type == "tool_result":
                        tool_id = item.get("tool_use_id")
                        if isinstance(tool_id, str) and tool_id in tool_uses:
                            tool_uses[tool_id]["result"] = text_from_content(item.get("content"))

    tool_counts = Counter(str(tool.get("name")) for tool in tools if tool.get("name"))
    bash_families: Counter[str] = Counter()
    bash_commands: list[str] = []
    read_files: set[str] = set()
    files_seen: set[str] = set()
    answer_filename_hits = 0
    answer_filename_input_hits = 0
    answer_file_target_hits = 0
    has_answer_hits = 0
    has_answer_input_hits = 0
    shell_errors = 0

    for tool in tools:
        name = str(tool.get("name") or "")
        tool_input = tool.get("input") if isinstance(tool.get("input"), dict) else {}
        input_text = json.dumps(tool_input, ensure_ascii=False)
        result = str(tool.get("result") or "")
        combined = input_text + "\n" + result
        files_seen.update(extract_session_files(combined))
        if "answer_" in combined:
            answer_filename_hits += 1
        if "answer_" in input_text:
            answer_filename_input_hits += 1
        if "has_answer" in combined:
            has_answer_hits += 1
        if "has_answer" in input_text:
            has_answer_input_hits += 1

        if name == "Bash":
            command = str(tool_input.get("command") or "")
            bash_commands.append(command)
            bash_families[command_family(command)] += 1
            if "answer_" in command:
                answer_file_target_hits += 1
            if "no matches found" in result or "is_error" in result:
                shell_errors += 1
        elif name == "Read":
            file_path = str(tool_input.get("file_path") or "")
            if "/session/" in file_path:
                read_files.add(Path(file_path).name)
            if "answer_" in file_path:
                answer_file_target_hits += 1

    return {
        "tool_counts": dict(tool_counts),
        "bash_families": dict(bash_families),
        "bash_commands": bash_commands,
        "tool_call_count": len(tools),
        "thinking_count": thinking_count,
        "read_session_file_count": len(read_files),
        "seen_session_file_count": len(files_seen),
        "answer_filename_hit_count": answer_filename_hits,
        "answer_filename_input_hit_count": answer_filename_input_hits,
        "answer_file_target_hit_count": answer_file_target_hits,
        "has_answer_hit_count": has_answer_hits,
        "has_answer_input_hit_count": has_answer_input_hits,
        "uses_answer_filename_signal": answer_filename_hits > 0,
        "actively_uses_answer_filename_signal": answer_filename_input_hits > 0,
        "targets_answer_file": answer_file_target_hits > 0,
        "uses_has_answer_signal": has_answer_hits > 0,
        "actively_uses_has_answer_signal": has_answer_input_hits > 0,
        "shell_error_count": shell_errors,
        "final_assistant_text": assistant_texts[-1] if assistant_texts else "",
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def judge_correct(path: Path) -> bool | None:
    if not path.exists():
        return None
    payload = load_json(path)
    judgement = payload.get("answer_judgement")
    if isinstance(judgement, dict) and isinstance(judgement.get("answer"), bool):
        return judgement["answer"]
    return None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counts: Counter[str] = Counter()
    bash_families: Counter[str] = Counter()
    for row in rows:
        tool_counts.update(row["tool_counts"])
        bash_families.update(row["bash_families"])

    correct_rows = [row for row in rows if row.get("correct") is True]
    incorrect_rows = [row for row in rows if row.get("correct") is False]

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(group),
            "tool_calls": summarize_numeric([int(row["tool_call_count"]) for row in group]),
            "read_session_files": summarize_numeric(
                [int(row["read_session_file_count"]) for row in group],
            ),
            "seen_session_files": summarize_numeric(
                [int(row["seen_session_file_count"]) for row in group],
            ),
            "answer_filename_signal_rate": (
                sum(1 for row in group if row["uses_answer_filename_signal"]) / len(group)
                if group
                else None
            ),
            "active_answer_filename_signal_rate": (
                sum(1 for row in group if row["actively_uses_answer_filename_signal"])
                / len(group)
                if group
                else None
            ),
            "answer_file_target_rate": (
                sum(1 for row in group if row["targets_answer_file"]) / len(group)
                if group
                else None
            ),
            "has_answer_signal_rate": (
                sum(1 for row in group if row["uses_has_answer_signal"]) / len(group)
                if group
                else None
            ),
            "active_has_answer_signal_rate": (
                sum(1 for row in group if row["actively_uses_has_answer_signal"]) / len(group)
                if group
                else None
            ),
        }

    return {
        "total": len(rows),
        "correct": len(correct_rows),
        "incorrect": len(incorrect_rows),
        "accuracy": len(correct_rows) / len(rows) if rows else None,
        "tool_counts": dict(tool_counts.most_common()),
        "bash_families": dict(bash_families.most_common()),
        "tool_calls": summarize_numeric([int(row["tool_call_count"]) for row in rows]),
        "read_session_files": summarize_numeric([int(row["read_session_file_count"]) for row in rows]),
        "seen_session_files": summarize_numeric([int(row["seen_session_file_count"]) for row in rows]),
        "answer_filename_signal_sessions": sum(
            1 for row in rows if row["uses_answer_filename_signal"]
        ),
        "active_answer_filename_signal_sessions": sum(
            1 for row in rows if row["actively_uses_answer_filename_signal"]
        ),
        "answer_file_target_sessions": sum(1 for row in rows if row["targets_answer_file"]),
        "has_answer_signal_sessions": sum(1 for row in rows if row["uses_has_answer_signal"]),
        "active_has_answer_signal_sessions": sum(
            1 for row in rows if row["actively_uses_has_answer_signal"]
        ),
        "correct_group": group_summary(correct_rows),
        "incorrect_group": group_summary(incorrect_rows),
    }


def markdown_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def fmt_num(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def fmt_rate(value: Any) -> str:
        if value is None:
            return "n/a"
        return f"{value * 100:.1f}%"

    lines = [
        "# Claude Code Session 分析",
        "",
        "本文基于 `longmemeval/{index}/claude_code_session.jsonl` 分析",
        "`longmemeval_session_answer` 为什么效果较好，以及实际使用了哪些工具。",
        "",
        "## 总体结果",
        "",
        f"- 样本数：{summary['total']}",
        f"- 正确：{summary['correct']}",
        f"- 错误：{summary['incorrect']}",
        f"- 准确率：{fmt_rate(summary['accuracy'])}",
        "",
        "## 工具使用",
        "",
        "| 工具 | 调用次数 |",
        "| --- | ---: |",
    ]
    for name, count in summary["tool_counts"].items():
        lines.append(f"| `{name}` | {count} |")

    lines.extend(["", "Bash 命令类型：", "", "| 类型 | 次数 |", "| --- | ---: |"])
    for name, count in summary["bash_families"].items():
        lines.append(f"| `{name}` | {count} |")

    tool_calls = summary["tool_calls"]
    read_files = summary["read_session_files"]
    seen_files = summary["seen_session_files"]
    lines.extend(
        [
            "",
            "## 搜索强度",
            "",
            "| 指标 | p50 | p90 | max | mean |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                "| 每个 query 的工具调用数 | "
                f"{fmt_num(tool_calls['p50'])} | {fmt_num(tool_calls['p90'])} | "
                f"{fmt_num(tool_calls['max'])} | {fmt_num(tool_calls['mean'])} |"
            ),
            (
                "| Read 直接读取的 session 文件数 | "
                f"{fmt_num(read_files['p50'])} | {fmt_num(read_files['p90'])} | "
                f"{fmt_num(read_files['max'])} | {fmt_num(read_files['mean'])} |"
            ),
            (
                "| 工具输入/输出中出现过的 session 文件数 | "
                f"{fmt_num(seen_files['p50'])} | {fmt_num(seen_files['p90'])} | "
                f"{fmt_num(seen_files['max'])} | {fmt_num(seen_files['mean'])} |"
            ),
            "",
            "## 潜在泄漏信号",
            "",
            f"- 出现 `answer_` 文件名信号的 session 数：{summary['answer_filename_signal_sessions']}",
            f"- 工具输入中主动引用 `answer_` 的 session 数：{summary['active_answer_filename_signal_sessions']}",
            f"- 直接把 `answer_` 文件作为 Read/grep 目标的 session 数：{summary['answer_file_target_sessions']}",
            f"- 出现 `has_answer` 字段信号的 session 数：{summary['has_answer_signal_sessions']}",
            f"- 工具输入中主动搜索/引用 `has_answer` 的 session 数：{summary['active_has_answer_signal_sessions']}",
            "",
            "这里的 `answer_` 指 session 文件名中包含 `@answer_xxx.json`；",
            "`has_answer` 指 session JSON 或工具输出中暴露了 `has_answer: true/false` 字段。",
            "这两个信号都可能显著降低任务难度，因为 agent 可以更快定位 gold session 或 gold message。",
            "",
            "## 正确和错误样本差异",
            "",
            "| 分组 | 数量 | 工具调用 mean | Read 文件 mean | 看见 answer_ | 主动 answer_ | 目标 answer 文件 | 看见 has_answer | 主动 has_answer |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ],
    )
    for key, label in [("correct_group", "正确"), ("incorrect_group", "错误")]:
        group = summary[key]
        lines.append(
            f"| {label} | {group['count']} | "
            f"{fmt_num(group['tool_calls']['mean'])} | "
            f"{fmt_num(group['read_session_files']['mean'])} | "
            f"{fmt_rate(group['answer_filename_signal_rate'])} | "
            f"{fmt_rate(group['active_answer_filename_signal_rate'])} | "
            f"{fmt_rate(group['answer_file_target_rate'])} | "
            f"{fmt_rate(group['has_answer_signal_rate'])} | "
            f"{fmt_rate(group['active_has_answer_signal_rate'])} |"
        )

    top_long_runs = sorted(rows, key=lambda row: int(row["tool_call_count"]), reverse=True)[:10]
    lines.extend(
        [
            "",
            "## 工具调用最多的样本",
            "",
            "| index | correct | tool_calls | Read session files | seen session files | answer_ | has_answer |",
            "| ---: | --- | ---: | ---: | ---: | --- | --- |",
        ],
    )
    for row in top_long_runs:
        lines.append(
            f"| {row['index']} | {row.get('correct')} | {row['tool_call_count']} | "
            f"{row['read_session_file_count']} | {row['seen_session_file_count']} | "
            f"{row['actively_uses_answer_filename_signal']} | {row['actively_uses_has_answer_signal']} |"
        )

    lines.extend(
        [
            "",
            "## 为什么效果好",
            "",
            "1. 它不是固定上下文回答，而是 agentic file search：先读 query，再用 `ls`、`grep`、`Read` 多轮定位证据。",
            "2. 它可以访问同一个 index 下完整的 `session/` 目录，因此召回空间比预先截断的 context 更大。",
            "3. Bash 搜索让模型能快速扫关键词、实体、日期、偏好词，而不需要把所有 session 塞进上下文。",
            "4. `Read` 工具让模型可以只精读命中的 session 文件，减少无关上下文干扰。",
            "5. 数据中存在 `answer_` 文件名和 `has_answer` 字段等强信号；如果 agent 看到或利用这些信号，准确率会被抬高，不应直接视为普通 RAG 能力。",
            "",
            "因此，90.4% 更适合作为“允许工具搜索完整 session 目录”的上界或诊断基线；",
            "如果要评估真实检索能力，需要移除 `answer_` 文件名、隐藏 `has_answer` 字段，并限制 agent 只能访问检索系统返回的候选内容。",
        ],
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze tool usage in copied LongMemEval Claude Code JSONL sessions.",
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--judge-name", default=DEFAULT_JUDGE_NAME)
    parser.add_argument("--json-output", default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--md-output", default=DEFAULT_MD_OUTPUT)
    parser.add_argument("--indices", default="", help="Comma-separated indices/ranges, e.g. 0,3,10-20.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = parse_indices(args.indices) if args.indices else None
    rows: list[dict[str, Any]] = []
    missing: list[int] = []

    for index_dir in iter_index_dirs(args.dataset_dir):
        index = int(index_dir.name)
        if selected is not None and index not in selected:
            continue
        session_path = index_dir / args.session_name
        if not session_path.exists():
            missing.append(index)
            continue
        row = parse_session(session_path)
        row.update(
            {
                "index": index,
                "session_path": str(session_path),
                "correct": judge_correct(index_dir / args.judge_name),
            },
        )
        rows.append(row)

    summary = aggregate(rows)
    summary["missing_session_indices"] = missing
    output = {"summary": summary, "rows": rows}

    json_path = args.dataset_dir / args.json_output
    md_path = args.dataset_dir / args.md_output
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(summary, rows), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
