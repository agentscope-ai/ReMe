"""Evaluation tools for ReMe LongMemEval benchmark."""

from pathlib import Path

import yaml

from reme.reme import ReMe


# Load prompts from YAML file
_YAML_PATH = Path(__file__).parent / "eval_reme.yaml"
with open(_YAML_PATH, "r", encoding="utf-8") as f:
    _PROMPTS = yaml.safe_load(f)


async def evaluation_for_memory_integrity(
    reme: ReMe,
    extract_memories: str,
    target_memory: str,
    model_name: str = "qwen3-max",
) -> dict:
    """
    Memory Integrity Evaluation

    Args:
        reme: ReMe instance
        extract_memories: A formatted string concatenating all memory points extracted by the memory system.
        target_memory: The target key memory point.
        model_name: Model name for evaluation

    Returns:
        dict with 'reasoning' and 'score' fields
    """
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY"].format(
        memories=extract_memories,
        expected_memory_point=target_memory,
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def evaluation_for_memory_accuracy(
    reme: ReMe,
    dialogue: str,
    golden_memories: str,
    candidate_memory: str,
    model_name: str = "qwen3-max",
) -> dict:
    """
    Memory Accuracy Evaluation

    Args:
        reme: ReMe instance
        dialogue: The complete human-machine dialogue record.
        golden_memories: The core memory points for this dialogue segment in the evaluation set .
        candidate_memory: A specific memory point extracted by the memory system being evaluated.
        model_name: Model name for evaluation

    Returns:
        dict with 'accuracy_score', 'is_included_in_golden_memories', and 'reason' fields
    """
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_MEMORY_ACCURACY"].format(
        dialogue=dialogue,
        golden_memories=golden_memories,
        candidate_memory=candidate_memory,
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def evaluation_for_update_memory(
    reme: ReMe,
    extract_memories: str,
    target_update_memory: str,
    original_memory: str,
    model_name: str = "qwen3-max",
) -> dict:
    """
    Memory Update Evaluation

    Args:
        reme: ReMe instance
        extract_memories: A formatted string concatenating all memory points extracted by the memory system .
        target_update_memory: The target updated memory point.
        original_memory: A formatted string concatenating all original memory points corresponding.
        model_name: Model name for evaluation

    Returns:
        dict with 'reason' and 'evaluation_result' fields
    """
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_UPDATE_MEMORY"].format(
        memories=extract_memories,
        updated_memory=target_update_memory,
        original_memory=original_memory,
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def evaluation_for_question(
    reme: ReMe,
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str,
    model_name: str = "qwen3-max",
) -> dict:
    """
    Question-Answering Evaluation

    Args:
        reme: ReMe instance
        question: The question string to be evaluated.
        reference_answer: The reference (gold-standard) answer.
        key_memory_points: The memory points used to derive the reference answer.
        response: The answer produced by the memory system.
        model_name: Model name for evaluation

    Returns:
        dict with 'reasoning' and 'evaluation_result' fields
    """
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION"].format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def evaluation_for_question2(
    reme: ReMe,
    question: str,
    reference_answer: str,
    key_memory_points: str,
    response: str,
    dialogue: str = "",
    model_name: str = "qwen3-max",
) -> dict:
    """
    Question-Answering Evaluation with Dialogue Context (Version 2)

    Args:
        reme: ReMe instance
        question: The question string to be evaluated.
        reference_answer: The reference (gold-standard) answer.
        key_memory_points: The memory points used to derive the reference answer.
        response: The answer produced by the memory system.
        dialogue: The formatted dialogue history (role, content, time_created).
        model_name: Model name for evaluation

    Returns:
        dict with 'reasoning' and 'evaluation_result' fields
    """
    prompt = _PROMPTS["EVALUATION_PROMPT_FOR_QUESTION2"].format(
        question=question,
        reference_answer=reference_answer,
        key_memory_points=key_memory_points,
        response=response,
        dialogue=dialogue if dialogue else "",
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result


async def answer_question_with_memories(
    reme: ReMe,
    question: str,
    memories: str,
    user_id: str = None,
    model_name: str = "qwen3-max",
) -> dict:
    """
    Answer a question using retrieved memories with PROMPT_MEMZERO_JSON template.

    Args:
        reme: ReMe instance
        question: The question to answer
        memories: The retrieved memories (formatted as context)
        user_id: Optional user ID for context formatting
        model_name: Model name for LLM request

    Returns:
        dict with 'reasoning' and 'answer' fields
    """
    # Format context with memories
    if user_id:
        context = _PROMPTS["TEMPLATE_MEMOS"].format(
            user_id=user_id,
            memories=memories,
        )
    else:
        context = f"Memories:\n{memories}"

    # Use PROMPT_MEMZERO_JSON template for structured JSON response
    prompt = _PROMPTS["PROMPT_MEMZERO_JSON"].format(
        context=context,
        question=question,
    )

    result = await reme.llm.simple_request_for_json(
        prompt=prompt,
        model_name=model_name,
    )

    return result
