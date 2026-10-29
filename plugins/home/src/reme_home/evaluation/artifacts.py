"""Safe evaluation artifacts, failure classification, and result summaries."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from uuid import uuid4

from ..preparation.data.models import CaseSpec, QuerySpec

from ..workspace import _validate_directory
from .compaction import round_scores


def contained(root: Path, value: str | Path, *, directory: bool = False) -> Path:
    """Reject traversal and symlinks instead of resolving through managed paths."""
    path = Path(value)
    if str(value).startswith("~") or ".." in path.parts:
        raise ValueError(f"Invalid artifact path: {value}")
    path = path if path.is_absolute() else root / path
    if not path.is_relative_to(root):
        raise ValueError(f"Artifact path escapes {root}: {value}")
    _validate_directory(root, path if directory else path.parent)
    if path.is_symlink():
        raise ValueError(f"Symbolic link at artifact: {path}")
    return path


def read_json(path: Path):
    """Read a regular persisted JSON artifact."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value, *, compact_leaf_containers: bool = False) -> None:
    """Atomically replace a rebuildable or progressively published result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".result-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            if compact_leaf_containers:
                stream.write(_dumps_compact_leaf_containers(value))
            else:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _dumps_compact_leaf_containers(value) -> str:
    """Pretty-print JSON while keeping small scalar-only containers on one line."""

    def is_atom(item) -> bool:
        return isinstance(item, (str, int, float, bool, type(None)))

    strings: set[str] = set()

    def collect_strings(item) -> None:
        if isinstance(item, str):
            strings.add(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                if isinstance(key, str):
                    strings.add(key)
                collect_strings(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect_strings(child)

    collect_strings(value)
    marker_prefix = f"\0meta-reme-singleton-{uuid4().hex}-"
    while any(item.startswith(marker_prefix) for item in strings):
        marker_prefix = f"\0meta-reme-singleton-{uuid4().hex}-"

    replacements: dict[str, str] = {}

    def replace_leaf_containers(item):
        if isinstance(item, dict):
            if len(item) == 1 and is_atom(next(iter(item.values()))):
                marker = f"{marker_prefix}{len(replacements)}"
                replacements[marker] = json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ": "),
                )
                return marker
            return {key: replace_leaf_containers(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            if 1 <= len(item) <= 2 and all(is_atom(child) for child in item):
                marker = f"{marker_prefix}{len(replacements)}"
                replacements[marker] = json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ": "),
                )
                return marker
            children = [replace_leaf_containers(child) for child in item]
            return children if isinstance(item, list) else tuple(children)
        return item

    transformed = replace_leaf_containers(value)
    content = json.dumps(transformed, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    for marker, replacement in replacements.items():
        encoded_marker = json.dumps(marker, ensure_ascii=False)
        content = content.replace(encoded_marker, replacement)
    return content


def file_sha256(path: Path) -> str:
    """Hash potentially large snapshots without loading them into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_archive(path: Path, max_bytes: int, destination: Path | None = None) -> None:
    """Validate every member before extracting regular files; never follow links."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing regular archive: {path}")
    if path.stat().st_size > max_bytes:
        raise ValueError(f"Archive exceeds byte limit: {path}")
    with tarfile.open(path, "r:gz") as archive:
        members = []
        seen = set()
        size = 0
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or name in seen:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            seen.add(name)
            size += max(0, member.size)
            if size > max_bytes or len(seen) > 100_000:
                raise ValueError("Archive exceeds extraction limits")
            members.append((member, name))
        if not any(member.isfile() for member, _ in members):
            raise ValueError("Archive contains no files")
        if destination is None:
            return
        destination.mkdir(parents=True, exist_ok=False)
        try:
            for member, name in members:
                target = destination.joinpath(*name.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)
        except BaseException:
            shutil.rmtree(destination)
            raise


def average_tokens(queries: list[dict]) -> dict:
    """Average candidate answer total tokens over returned queries, like Meta-ReMe."""
    totals = {}
    for query in queries:
        usage_by_model = (query.get("answer_result") or {}).get("token_usage", {})
        if not isinstance(usage_by_model, dict):
            continue
        for model, usage in usage_by_model.items():
            if not isinstance(usage, dict):
                continue
            total = usage.get("total_tokens")
            if isinstance(total, int) and not isinstance(total, bool) and total != 0:
                totals[model] = totals.get(model, 0) + total
    return {model: round(total / len(queries), 3) for model, total in totals.items()} if queries else {}


def summarize(run: Path, cases: list, *, scoring_required: bool = True) -> dict:
    """Rebuild compact optimizer evidence with Meta-ReMe's expected-query denominator."""
    rows, all_scores, query_results = {}, [], []
    case_counts = {"completed": 0, "candidate_failure": 0, "infra_error": 0}
    construction_counts = {"success": 0, "partial_success": 0, "failure": 0}
    query_counts = {"completed": 0, "candidate_failure": 0, "infra_error": 0}
    stages = {"construction": 0, "query": 0}
    expected = sum(len(case.queries) for case in cases)
    for case in cases:
        folder = contained(run, f"cases/{case.case_id}", directory=True)
        construction_path = contained(folder, "memory_construction/result.json")
        construction = read_json(construction_path) if construction_path.is_file() else {}
        query_path = contained(folder, "queries/result.json")
        queries = read_json(query_path).get("queries", []) if query_path.is_file() else []
        failure = read_json(folder / "failure.json") if (folder / "failure.json").is_file() else {}
        if not construction and not queries and not failure:
            continue
        scores = [query["score"] for query in queries if query.get("score") is not None]
        all_scores.extend(scores)
        query_results.extend(queries)
        failed = sum(query.get("error") is not None for query in queries)
        has_infra = failure.get("failure_kind") == "infra_error" or any(q.get("infrastructure_error") for q in queries)
        case_status = (
            "infra_error"
            if has_infra
            else "candidate_failure" if failed or failure or not construction else "completed"
        )
        case_counts[case_status] += 1
        construction_status = construction.get("status", "failure")
        if construction_status == "reused":
            construction_status = "success"
        construction_counts[construction_status] += 1
        jobs = [job for job in construction.get("jobs", []) if isinstance(job, dict)]
        job_results = [job["result"] if isinstance(job.get("result"), dict) else {} for job in jobs]
        construction_errors = sum(not result.get("success") for result in job_results)
        stages["construction"] += construction_status == "failure" or bool(construction_errors)
        stages["query"] += bool(failed)
        compact_queries, query_errors = {}, {}
        for query in queries:
            query_status = (
                "completed"
                if query.get("error") is None
                else "infra_error" if query.get("infrastructure_error") else "candidate_failure"
            )
            query_counts[query_status] += 1
            compact_queries[query["query_id"]] = {
                "question_type": query.get("question_type"),
                "score": query.get("score"),
            }
            if query_status != "completed":
                error = str(query["error"])
                query_errors[query["query_id"]] = {
                    "failure_kind": query_status,
                    "stage": query.get("failure_stage"),
                    "error": error if len(error) <= 500 else error[:500] + "…",
                }
        rows[case.case_id] = {
            "construction": {
                "status": construction_status,
                "success_job_count": len(jobs) - construction_errors,
                "fail_job_count": construction_errors,
            },
            "mean_score": None if has_infra or not scoring_required else sum(scores) / len(case.queries),
            "average_query_total_token_usage": average_tokens(queries),
            "queries": compact_queries,
        }
        if failure.get("stage") is not None:
            rows[case.case_id]["stage"] = failure["stage"]
        if construction_errors:
            rows[case.case_id]["construction_error_count"] = construction_errors
        if failed:
            rows[case.case_id]["query_error_count"] = failed
            rows[case.case_id]["query_errors"] = query_errors
    groups = {}
    for query in query_results:
        groups.setdefault(str(query.get("question_type") or "unknown"), []).append(query)
    statistics = {}
    for name, queries in sorted(groups.items()):
        scores = [query["score"] for query in queries if query.get("score") is not None]
        statistics[name] = {
            "query_count": len(queries),
            "average_score": sum(scores) / len(scores) if scores else None,
            "average_query_total_token_usage": average_tokens(queries),
        }
    return round_scores(
        {
            "case_count": len(rows),
            "scoring_enabled": scoring_required,
            "case_status_counts": case_counts,
            "construction_status_counts": construction_counts,
            "failure_stage_case_counts": stages,
            "expected_query_count": expected,
            "scored_query_count": len(all_scores),
            "query_status_counts": {
                **query_counts,
                "unscored": max(expected - len(all_scores), 0) if scoring_required else 0,
            },
            "mean_query_score": (
                sum(all_scores) / expected if scoring_required and expected and not case_counts["infra_error"] else None
            ),
            "observed_mean_query_score": sum(all_scores) / len(all_scores) if all_scores else None,
            "low_score_cases": [
                {key: row["mean_score"]}
                for key, row in sorted(
                    rows.items(),
                    key=lambda item: (item[1]["mean_score"] if item[1]["mean_score"] is not None else 2, item[0]),
                )
                if row["mean_score"] is not None
                and any(query.get("score") is not None and query["score"] < 1 for query in row["queries"].values())
            ],
            "question_type_statistics": statistics,
            "cases": rows,
        },
    )


def rebuild_summary(run: Path) -> dict:
    """Rebuild compact evidence from the manifest and authoritative phase results."""
    manifest = read_json(run / "manifest.json")
    cases = [
        CaseSpec(
            case_id=case_id,
            sessions=[],
            queries=[QuerySpec(query_id=query_id, question="", golden_answer=None) for query_id in query_ids],
        )
        for case_id, query_ids in manifest["query_ids"].items()
    ]
    by_id = {case.case_id: case for case in cases}
    if manifest.get("status") == "completed":
        for case_id in manifest["case_ids"]:
            folder = contained(run, f"cases/{case_id}", directory=True)
            read_json(contained(folder, "memory_construction/result.json"))
            query_path = contained(folder, "queries/result.json")
            queries = read_json(query_path)["queries"] if query_path.is_file() else []
            actual_ids = [query["query_id"] for query in queries]
            if actual_ids != manifest["query_ids"][case_id]:
                if queries or not (folder / "failure.json").is_file():
                    raise ValueError(f"Completed evaluation has incomplete query results: {case_id}")
    summary = summarize(
        run,
        [by_id[case_id] for case_id in manifest["case_ids"]],
        scoring_required=manifest.get("mode") != "debug",
    )
    name = "summary.json" if manifest.get("status") == "completed" else "partial_summary.json"
    write_json(run / name, summary, compact_leaf_containers=True)
    return summary


INFRA_TYPES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectionError",
    "ConnectTimeout",
    "NetworkError",
    "PoolTimeout",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "ServiceUnavailableError",
    "TimeoutError",
    "TimeoutException",
    "WriteError",
    "WriteTimeout",
    "SandboxCommandTimeoutError",
}


def failure_kind(exception, stage: str) -> str:
    """Only candidate execution errors count against candidate scores."""
    if stage not in {"construction", "answer"}:
        return "infra_error"
    exception = exception if isinstance(exception, dict) else {}
    if not isinstance(exception.get("type"), str) or not exception["type"]:
        return "infra_error"  # Legacy/unknown failures cannot support a fair comparison.
    causes = exception.get("causes", [])
    if isinstance(causes, list) and any(failure_kind(cause, stage) == "infra_error" for cause in causes):
        return "infra_error"
    status = exception.get("status_code")
    if isinstance(status, int) and (status in {408, 429} or status >= 500):
        return "infra_error"
    name = str(exception.get("type", ""))
    module = str(exception.get("module", ""))
    if name in INFRA_TYPES or (
        module.startswith(("httpx", "httpcore", "aiohttp", "requests", "urllib3"))
        and any(word in name.lower() for word in ("connection", "network", "timeout", "protocol"))
    ):
        return "infra_error"
    return "candidate_failure"


def _failure_exception(exceptions: list, stage: str):
    """Infrastructure or unclassified failures take priority over code errors."""
    return next(
        (exception for exception in exceptions if failure_kind(exception, stage) == "infra_error"),
        exceptions[0] if exceptions else None,
    )


def construction_exception(build: dict):
    """Inspect every failed Job so an earlier code error cannot hide an outage."""
    exceptions = []
    for job in build.get("jobs", []):
        result = job.get("result")
        if not isinstance(result, dict):
            exceptions.append(None)
        elif not result.get("success"):
            exceptions.append(result.get("exception"))
    if build.get("exception"):
        exceptions.append(build["exception"])
    return _failure_exception(exceptions, "construction")


def answer_exception(queries: dict):
    """Select the representative error from failed answer results."""
    return _failure_exception(
        [row.get("exception") for row in queries.get("queries", []) if row.get("error")],
        "answer",
    )
