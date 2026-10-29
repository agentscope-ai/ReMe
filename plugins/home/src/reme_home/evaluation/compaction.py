"""Result compaction rules preserved from dev_meta b53048e4.

Keep diagnostic evidence while removing reconstructable prompts and duplicate data.
Compaction copies inputs so scoring and failure classification retain raw responses.
"""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any


def round_scores(value: Any, *, score: bool = False) -> Any:
    """Copy structured scores with four decimal places, preserving text and other numbers."""
    if isinstance(value, dict):
        return {
            key: round_scores(
                child,
                score=score
                or isinstance(key, str)
                and (
                    key in {"score", "precision", "recall", "f1", "tau_norm", "low_score_cases"}
                    or key.endswith("_score")
                ),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [round_scores(child, score=score) for child in value]
    return round(value, 4) if score and isinstance(value, float) else value


def compact_query(result: dict) -> dict:
    """Copy and compact one authoritative query result."""
    result = deepcopy(result)
    _compact_query_result(result)
    # Reference answers and provider raw text remain source evidence.
    for key in ("score", "judge_result", "answer_result"):
        if key in result:
            result[key] = round_scores(result[key], score=key == "score")
    return result


def compact_construction(build: dict) -> dict:
    """Copy and compact ordered construction jobs using the reviewed rules."""
    persisted_build = deepcopy(build)
    _compact_construction_source_conversations(persisted_build)
    _compact_construction_index_snapshots(persisted_build)
    for job in persisted_build.get("jobs", []):
        if not isinstance(job, Mapping):
            continue
        if isinstance(job, dict):
            job.pop("memory_checkpoint", None)
        result = job.get("result")
        if not isinstance(result, dict):
            continue
        _compact_outer_job_call_count(job.get("job"), result)
        token_usage = result.get("token_usage")
        if not isinstance(token_usage, dict):
            continue
        for backend, usage in list(token_usage.items()):
            if isinstance(usage, Mapping) and all(
                usage.get(field) is None for field in ("input_tokens", "output_tokens", "total_tokens")
            ):
                del token_usage[backend]
        if not token_usage:
            del result["token_usage"]
        elif all(
            isinstance(usage, Mapping)
            and all(usage.get(field) == 0 for field in ("input_tokens", "output_tokens", "total_tokens"))
            for usage in token_usage.values()
        ):
            result["token_usage"] = 0
    return persisted_build


def _compact_query_result(result: dict[str, Any]) -> None:
    """Remove reconstructable prompts and uninformative token records before persistence."""

    answer_result = result.get("answer_result")
    if isinstance(answer_result, dict):
        if answer_result.get("answer") == result.get("answer"):
            answer_result.pop("answer", None)
        if "error" in result and answer_result.get("error") == result.get("error"):
            answer_result.pop("error", None)
        _compact_outer_job_call_count("agentic_answer", answer_result)
        metadata = answer_result.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("query") == result.get("question"):
                metadata.pop("query", None)
            if metadata.get("response") == result.get("answer"):
                metadata.pop("response", None)

    judge_result = result.get("judge_result")
    if isinstance(judge_result, dict):
        judge_result.pop("token_usage", None)
        if result.get("score") is not None:
            judge_result.pop("answer", None)
        _compact_outer_job_call_count("answer_judge", judge_result)
        metadata = judge_result.get("metadata")
        if isinstance(metadata, dict):
            if metadata.get("raw_answer_judgement") is not None:
                metadata.pop("answer_judgement", None)
            for field in (
                "agent_answer",
                "golden_answer",
                "llm_response",
                "probing_question",
                "query",
                "question_type",
                "rubric",
            ):
                metadata.pop(field, None)

    def compact(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("sys_prompt", None)
            token_usage = value.get("token_usage")
            if isinstance(token_usage, dict):
                for backend, usage in list(token_usage.items()):
                    if isinstance(usage, Mapping) and all(
                        usage.get(field) is None for field in ("input_tokens", "output_tokens", "total_tokens")
                    ):
                        del token_usage[backend]
                if not token_usage:
                    del value["token_usage"]
            for child in value.values():
                compact(child)
        elif isinstance(value, list):
            for child in value:
                compact(child)

    compact(result)


def _compact_outer_job_call_count(job_name: Any, result: dict[str, Any]) -> None:
    """Remove one call already represented by the enclosing result record."""

    counts = result.get("job_call_counts")
    if not isinstance(job_name, str) or not isinstance(counts, dict):
        return
    own_count = counts.get(job_name)
    if isinstance(own_count, int) and not isinstance(own_count, bool) and own_count > 0:
        remaining = own_count - 1
        if remaining:
            counts[job_name] = remaining
        else:
            del counts[job_name]
    if not counts:
        del result["job_call_counts"]


def _compact_construction_index_snapshots(value: Any) -> None:
    """Replace cumulative daily-index note snapshots with their counts."""

    if isinstance(value, dict):
        index = value.get("index")
        if _is_daily_index_snapshot(index):
            index["notes_count"] = len(index.pop("notes"))
        for child in value.values():
            _compact_construction_index_snapshots(child)
    elif isinstance(value, list):
        for child in value:
            _compact_construction_index_snapshots(child)


def _is_daily_index_snapshot(value: Any) -> bool:
    """Recognize the stable ``refresh_day_index`` payload without assuming a configured directory name."""

    if not isinstance(value, dict) or not isinstance(value.get("notes"), list):
        return False
    date = value.get("date")
    path = value.get("path")
    if not isinstance(date, str) or not isinstance(path, str):
        return False
    index_path = PurePosixPath(path)
    return (
        index_path.suffix == ".md"
        and index_path.stem == date
        and isinstance(value.get("created"), bool)
        and isinstance(value.get("changed"), bool)
    )


def _compact_construction_source_conversations(value: Any) -> None:
    """Drop a transcript link when its sibling session ID already identifies the same file."""

    if isinstance(value, dict):
        session_id = value.get("session_id")
        source_conversation = value.get("source_conversation")
        if isinstance(session_id, str) and isinstance(source_conversation, str):
            source_path = source_conversation.strip()
            if source_path.startswith("[[") and source_path[-2:] == "]]":
                source_path = source_path[2:-2]
            if PurePosixPath(source_path).name == f"{session_id}.jsonl":
                del value["source_conversation"]
        for child in value.values():
            _compact_construction_source_conversations(child)
    elif isinstance(value, list):
        for child in value:
            _compact_construction_source_conversations(child)
