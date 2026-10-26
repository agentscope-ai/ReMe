"""MemLens's task-specific binary judge for fixed saved predictions.

Prompts, answer preprocessing and verdict parsing follow SOURCE_REVISION.
Copyright (c) 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors.
The complete MIT notice accompanies the verbatim _memlens_judge_prompts.json.
Only exhausted API errors differ: they remain coverage failures, not wrong answers.
"""

from functools import lru_cache
import json
from pathlib import Path
import re

SOURCE_REVISION = "77f3ab9a52fa2d6a17978e2dffe80438a4ecced2"

PROTOCOL_ID = "memlens-task-binary-api-qwen-v1"
_HEDGING_MARKERS = (
    "wait",
    "let me re-read",
    "let me re read",
    "let me double check",
    "let me double-check",
    "let me reconsider",
    "let me look again",
    "let's reconsider",
    "let s reconsider",
    "re-evaluating",
    "re evaluating",
    "actually looking",
    "actually let me",
    "hold on",
    "let me look at",
    "let me check",
    "let's re-read",
    "let s re-read",
    "i need to reconsider",
    "wait let me",
)


@lru_cache(maxsize=1)
def _prompts() -> dict:
    """Load the checked-in, verbatim official prompt strings, never a network resource."""
    return json.loads(Path(__file__).with_name("_memlens_judge_prompts.json").read_text(encoding="utf-8"))


def task_key(category: str, subtype: str, reference: str) -> str:
    """Official question-type/subtype routing, including reference-based MSR choices."""
    if category == "multi_session_reasoning":
        if subtype == "arithmetic":
            return "MSR_Arithmetic"
        if subtype == "entity_resolution" and reference.strip() in {"Yes", "No"}:
            return "MSR_YesNo"
        return "MSR_Counting"
    if category == "temporal_reasoning":
        return {
            "duration_comparison": "TR_DurationComparison",
            "order_ranking": "TR_OrderRanking",
        }.get(subtype, "TR_DateExtraction")
    fixed = {"knowledge_update": "KU_KnowledgeUpdate", "answer_refusal": "AR_AnswerRefusal"}
    if category in fixed:
        return fixed[category]
    if category == "information_extraction":
        return "IE_PreviousInfo" if subtype == "previnfo" else "IE_Entity"
    raise ValueError(f"Unsupported MemLens judge category: {category}")


def judge_prompt(
    question: str,
    reference: str,
    prediction: str,
    category: str,
    subtype: str = "",
    old_answer: str = "",
) -> str:
    """Render the exact official rubric/examples and current-case fields."""
    if any(not isinstance(value, str) for value in (question, reference, prediction, category, subtype, old_answer)):
        raise ValueError("MemLens judge inputs must be strings")
    key = task_key(category, subtype, reference)
    prompts = _prompts()
    prompt = prompts["template"].format(task_criteria=prompts["criteria"][key], examples=prompts["examples"][key])
    if key == "AR_AnswerRefusal":
        reference = "Insufficient information"
    old = f"<Old (Outdated) Answer>: {old_answer}\n" if key == "KU_KnowledgeUpdate" and old_answer else ""
    return (
        f"{prompt}[Current Case]\n<Question>: {question}\n<Standard Answer>: {reference}\n"
        f"{old}<Student Answer>: {prediction}\n\n[Scoring Rationale]:"
    )


def normalize_for_judge(text: str) -> dict:
    """Preserve native parsing, circularity diagnostics and final-30% tail selection."""
    if not text:
        return {"text": "", "original_len": 0, "was_truncated": False, "is_degenerate": False, "diagnostics": {}}
    original_len = len(text)
    text = parse_model_output(text)
    hedging, ratio = 0, 1.0
    if len(text) >= 500:
        lower = text.lower()
        hedging = sum(lower.count(marker) for marker in _HEDGING_MARKERS)
        words = lower.split()
        ngrams = [tuple(words[index : index + 5]) for index in range(len(words) - 4)]
        ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    truncated = len(text) > 3000 and len(text) > 6000
    if truncated:
        text = "[...truncated earlier reasoning...]\n" + text[int(len(text) * (1 - 0.3)) :]
    return {
        "text": text,
        "original_len": original_len,
        "was_truncated": truncated,
        "is_degenerate": hedging >= 6 or ratio < 0.40,
        "diagnostics": {"hedging_count": hedging, "unique_5gram_ratio": round(ratio, 3)},
    }


def parse_judge_response(text: str) -> dict:
    """Official first-match priority/clamping; even unparseable outputs natively score zero."""
    patterns = (
        (r'"answer_score"\s*:\s*(\d+)', 0),
        (r"deserves\s+(\d+)\s+point", re.IGNORECASE),
        (r"\[Score\]\s*:\s*(\d+)", 0),
    )
    for pattern, flags in patterns:
        match = re.search(pattern, text, flags) if text else None
        if match:
            return {"acc": float(min(int(match.group(1)), 1)), "parse_default_zero": False}
    return {"acc": 0.0, "parse_default_zero": True}


def parse_model_output(text: str) -> str:
    """Native special-token/think-trace parsing; retained raw outputs remain separate."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<\|[^>]*\|>", "", text)
    text = re.sub(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", r"\1", text, flags=re.DOTALL)
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    return text.replace("[CONTENT_FILTER_REJECTED]", "").strip()
