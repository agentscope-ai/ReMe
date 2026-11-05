"""Unchanged MobileMem/MemBase judge prompts and binary parser.

Adapted from ZJUNLP's MobileMem and MemBase revisions below.
MIT License, Copyright (c) 2026 ZJUNLP; see LICENSE-CODE.
"""

import json
import re
from string import Template

SOURCE_REVISION = "fa5cd13c9c659699df9caf650987ba2bea8ca085"
MEMBASE_REVISION = "a9ba9495d48541d3e68588ff097294cc7a1201bc"
PROTOCOL_ID = "mobilemem-omni-membase-default-judge-v1"
STRICT_PROTOCOL_ID = "mobilemem-omni-final-memory-native-v1"
TEMPLATE_NAME = "mobilemem-omni-judge"
SOURCE_URL = f"https://github.com/zjunlp/MemBase/blob/{MEMBASE_REVISION}/membase/inference_utils/prompts.py"
JUDGE_TEMPLATE = (
    "Your task is to classify an answer to a question as CORRECT or WRONG. "
    "You are given a question, a gold answer, a generated answer, and optional "
    "evidence from the user's prior conversations.\n\n"
    "GRADING RULES:\n"
    "1. The generated answer must include every key element in the gold answer.\n"
    "2. Unsupported or irrelevant content that changes the meaning makes the answer WRONG.\n"
    "3. Equivalent date formats are acceptable when they refer to the same time.\n"
    "4. Use the evidence only to verify the answer; do not infer unsupported facts.\n\n"
    "Question: $question\n"
    "Gold answer: $golden_answers\n"
    "Evidence: $evidence\n"
    "Generated answer: $prediction\n\n"
    'Return exactly one JSON object: {"label": "CORRECT"} or '
    '{"label": "WRONG"}.'
)
STRICT_JUDGE_TEMPLATE = (
    "Your task is to classify an answer to a question as CORRECT or WRONG. "
    "You will be provided with the following information:\n"
    "(1) A question (asked by a user to an AI assistant);\n"
    "(2) A gold (ground truth) answer;\n"
    "(3) A generated answer;\n"
    "(4) A set of evidence sentences relevant to the question, which may be used to verify the generated answer.\n"
    "The question focuses on information that an AI assistant should recall about the user "
    "based on prior conversations. "
    "The gold answer is typically concise and includes all key facts.\n\n"
    "GRADING RULES (Must Be Strictly Followed)\n"
    "1. COMPLETE COVERAGE REQUIRED: The generated answer must include every key element present in the gold answer. "
    "Missing any required fact results in a WRONG classification.\n"
    "2. NO REDUNDANT CONTENT: Including excessive or irrelevant details not supported by the evidence or gold answer "
    "results in a WRONG classification.\n"
    "3. TIME-RELATED QUESTIONS: Different date formats (e.g., May 7th vs. 7 May) are acceptable "
    "as long as the same date or time period is referenced.\n\n"
    "INSTRUCTIONS FOR THIS TASK\n"
    "Use the provided evidence only as supporting context for verification when available. "
    "Do not infer facts that are not present in the question, gold answer, or evidence.\n"
    "First, provide a brief (one-sentence) explanation of your reasoning. Then, output the label CORRECT or WRONG.\n"
    "Do not include both labels in your response, as this will break the evaluation script.\n"
    "Return the label in JSON format with the key label.\n\n"
    "QUESTION TO EVALUATE\n"
    "Question: [Question]\n"
    "Gold answer: [Gold Answer]\n"
    "Evidence: [Evidences]\n"
    "Generated answer: [Generated Answer]"
)


def protocol_id(judge_protocol: str) -> str:
    """Identify the selected rubric without silently substituting another."""
    if judge_protocol == "membase_default":
        return PROTOCOL_ID
    if judge_protocol == "native_strict":
        return STRICT_PROTOCOL_ID
    raise ValueError(f"Unsupported MobileMem judge protocol: {judge_protocol!r}")


def judge_prompt(
    question: str,
    reference: str,
    prediction: str,
    evidence=(),
    *,
    judge_protocol="membase_default",
) -> str:
    """Preserve native single-pass substitution and untrimmed evidence."""
    protocol_id(judge_protocol)
    evidence_text = "\n".join(str(item) for item in evidence) or "Not provided."
    if judge_protocol == "membase_default":
        return Template(JUDGE_TEMPLATE).substitute(
            question=question,
            golden_answers=reference,
            evidence=evidence_text,
            prediction=prediction,
        )
    replacements = {
        "[Question]": question,
        "[Gold Answer]": reference,
        "[Evidences]": evidence_text,
        "[Generated Answer]": prediction,
    }
    return re.sub(
        r"\[Question\]|\[Gold Answer\]|\[Evidences\]|\[Generated Answer\]",
        lambda match: replacements[match[0]],
        STRICT_JUDGE_TEMPLATE,
    )


def parse_judge(text: str) -> dict:
    """Preserve MemBase's WRONG-first binary parsing and native default zero."""
    text = text.strip()
    try:
        parsed = json.loads(text)
        label = str(parsed.get("label", ""))
    except (json.JSONDecodeError, AttributeError):
        label = text
    normalised = label.strip().upper()
    score = 0.0 if "WRONG" in normalised else float("CORRECT" in normalised)
    return {
        "llm_judge_score": score,
        "label": "CORRECT" if score else "WRONG",
        "parse_default_zero": "WRONG" not in normalised and "CORRECT" not in normalised,
    }
