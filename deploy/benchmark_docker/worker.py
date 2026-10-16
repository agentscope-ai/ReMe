"""In-container worker that calls ReMe jobs directly, without HTTP.

The host uploads this file into every case sandbox. It deliberately imports
ReMe only after process startup so source candidates can control exactly which
ReMe package is loaded.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any

_WORKSPACE_PATH_FIELDS = (
    "metadata_dir",
    "session_dir",
    "mem_session_dir",
    "resource_dir",
    "daily_dir",
    "digest_dir",
    "dialog_dir",
)
_ANALYSIS_EXCLUDE_FIELDS = ("metadata_dir", "resource_dir")


def _exception_details(exc: BaseException) -> dict[str, Any]:
    """Return small, structured exception metadata for host-side classification."""

    details: dict[str, Any] = {
        "module": type(exc).__module__,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        details["status_code"] = status_code
    error_number = getattr(exc, "errno", None)
    if isinstance(error_number, int):
        details["errno"] = error_number
    return details


def _write_runtime_layout(case_root: Path, app_config: Any) -> None:
    """Persist validated, non-secret paths needed by selective export."""
    resolved_case_root = case_root.resolve()
    workspace_root = Path(app_config.workspace_dir).resolve()
    try:
        workspace_relative = workspace_root.relative_to(resolved_case_root)
    except ValueError as exc:
        raise ValueError(f"sandbox workspace must stay under {resolved_case_root}: {workspace_root}") from exc

    resolved_paths: dict[str, str | None] = {}
    for field in _WORKSPACE_PATH_FIELDS:
        configured = (
            str(Path(app_config.session_dir) / "dialog")
            if field == "dialog_dir"
            else str(getattr(app_config, field)).strip()
        )
        if not configured:
            resolved_paths[field] = None
            continue
        path = (workspace_root / configured).resolve()
        try:
            relative = path.relative_to(resolved_case_root)
        except ValueError as exc:
            raise ValueError(f"sandbox config path {field} must stay under {workspace_root}: {path}") from exc
        if path != workspace_root and workspace_root not in path.parents:
            raise ValueError(f"sandbox config path {field} must stay under {workspace_root}: {path}")
        resolved_paths[field] = relative.as_posix()

    analysis_excludes = list(
        dict.fromkeys(path for field in _ANALYSIS_EXCLUDE_FIELDS if (path := resolved_paths[field]) is not None),
    )
    layout = {
        "workspace_root": workspace_relative.as_posix(),
        "configured_paths": resolved_paths,
        "analysis_excludes": analysis_excludes,
    }
    (resolved_case_root / "runtime-layout.json").write_text(
        json.dumps(layout, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_runtime(request: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Resolve one request's config after confining process-local output."""
    case_root = Path(request["case_root"])
    case_tmp = case_root / "tmp"
    case_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(case_tmp)
    os.chdir(case_root)

    # Import after setting TMPDIR so ReMe and its subprocesses cannot leave
    # case-specific temporary files outside the disposable case root.
    from reme.config import resolve_app_config
    from reme.plugin import resolve_plugin_runtime
    from reme.schema import ApplicationConfig

    plugins = request.get("plugins") or []
    if not isinstance(plugins, list) or not all(isinstance(name, str) and name for name in plugins):
        raise TypeError("sandbox request plugins must be a list of non-empty strings")
    app_config = resolve_app_config(
        config=request.get("config") or "benchmark.yaml",
        plugins=plugins,
        workspace_dir=request["workspace_dir"],
        log_to_console=False,
        log_to_file=True,
    )
    app_config["environment"] = dict(os.environ)
    # Apply plugin defaults before validation materializes schema defaults,
    # otherwise those defaults would override plugin paths and partial jobs.
    runtime = resolve_plugin_runtime(app_config)
    resolved_config = ApplicationConfig.model_validate(runtime.config)
    _write_runtime_layout(case_root, resolved_config)
    # The build checkpoint reads workspace paths from this mapping. Return the
    # validated model's dump so optional config fields (including ``daily_dir``)
    # have the same defaults as the Application instance.
    return case_root, resolved_config.model_dump()


async def _run_job_on_app(app: Any, job: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one job and return token and ReMe job invocation deltas."""
    from reme.enumeration import ComponentEnum
    from reme.utils.evaluation_interface import track_agent_token_usage, track_job_counts

    agent_names = list(app.context.components.get(ComponentEnum.AGENT_WRAPPER, {}))
    job_names = list(app.context.jobs)
    token_usage: dict[str, dict[str, int | None]] = {}
    job_call_counts: dict[str, int] = {}
    try:
        with (
            track_agent_token_usage(agent_names, app.context) as token_usage,
            track_job_counts(job_names, app.context) as job_call_counts,
        ):
            response = await app.run_job(job, **arguments)
    except Exception as exc:  # Preserve usage accumulated before a job failure.
        result = {
            "success": False,
            "answer": "",
            "metadata": {},
            "error": f"{type(exc).__name__}: {exc}",
            "exception": _exception_details(exc),
            "traceback": traceback.format_exc(),
        }
    else:
        result = {
            "success": bool(response.success),
            "answer": response.answer,
            "metadata": response.metadata,
            "error": None if response.success else str(response.answer),
        }
    result["token_usage"] = token_usage
    result["job_call_counts"] = {name: count for name, count in job_call_counts.items() if count}
    return result


def _atomic_write_json(path: Path, value: Any) -> None:
    """Publish one JSON artifact without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _start_case_log(path: Path) -> Any:
    """Attach an exact artifact path to ReMe's configured logger backend."""
    from reme.utils import get_logger

    path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_to_console=False, log_to_file=False, force_init=True)
    if hasattr(logger, "add"):  # Loguru logger.
        logger.add(str(path), level="INFO", encoding="utf-8")
    else:  # Standard library logger, selected by REME_DISABLE_LOGURU.
        logger.addHandler(logging.FileHandler(path, encoding="utf-8"))
    return logger


def _validate_query_id(query_id: Any) -> str:
    """Validate an ID used verbatim as a directory name in the artifact."""
    if not isinstance(query_id, str) or not query_id or len(query_id.encode("utf-8")) > 255:
        raise ValueError("query_id must be a non-empty string no longer than 255 UTF-8 bytes")
    if query_id in {".", ".."} or "/" in query_id or "\\" in query_id or "\x00" in query_id:
        raise ValueError(f"query_id is not a safe directory name: {query_id!r}")
    if query_id == "summary.json":
        raise ValueError("query_id conflicts with the queries summary: 'summary.json'")
    return query_id


def _clear_query_state(app: Any) -> None:
    """Drop request-scoped metadata that must not leak into the next query."""
    app.context.metadata.pop("tool_contexts", None)
    app.context.metadata.pop("__search_call_budgets", None)


@asynccontextmanager
async def _application_scope(app: Any, failed_results: list[dict[str, Any]] | None = None) -> AsyncIterator[None]:
    """Close the app, attaching cleanup diagnostics to already-failed results.

    Propagating exceptions retain cleanup notes; cleanup alone still raises.
    """
    error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            await app.close()
        except BaseException as exc:
            if error is None:
                if isinstance(exc, Exception) and failed_results:
                    diagnostic = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "exception": _exception_details(exc),
                        "traceback": traceback.format_exc(),
                    }
                    for result in failed_results:
                        result.setdefault("cleanup_errors", []).append(diagnostic)
                else:
                    raise
            else:
                error.add_note("Application close failed:\n" + "".join(traceback.format_exception(exc, chain=False)))


async def _run(request: dict[str, Any]) -> dict[str, Any]:
    """Construct an Application, run one job, and close it deterministically."""
    job = request["job"]
    job_args = request.get("arguments") or {}
    _, app_config = _prepare_runtime(request)
    from reme.reme import ReMe

    app = ReMe(**app_config)
    failed_results: list[dict[str, Any]] = []
    async with _application_scope(app, failed_results):
        await app.start()
        result = await _run_job_on_app(app, job, job_args)
        if not result["success"]:
            failed_results.append(result)
    return result


def _append_build_error(build_log: Path, detail: str) -> None:
    """Keep build diagnostics even when the candidate reconfigures the logger."""
    with build_log.open("a", encoding="utf-8") as log_file:
        log_file.write(detail.rstrip() + "\n")


async def _run_build(request: dict[str, Any]) -> dict[str, Any]:
    """Execute all construction jobs in one Application and one build log."""
    build_log = Path(request["case_root"]) / "build_log" / "build.log"
    if build_log.exists():
        raise FileExistsError(f"build artifact already exists: {build_log}")
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.touch()

    try:
        case_root, app_config = _prepare_runtime(request)
        from reme.reme import ReMe
        from reme.utils import get_logger

        jobs = request.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("build request requires a non-empty jobs list")
        app = ReMe(**app_config)
        logger = None
        results: list[dict[str, Any]] = []
        failed_results: list[dict[str, Any]] = []
        try:
            async with _application_scope(app, failed_results):
                # Application.__init__ resets the global logger. Attach afterwards.
                logger = _start_case_log(build_log)
                logger.info("[sandbox] build phase started")
                await app.start()
                for index, specification in enumerate(jobs):
                    if not isinstance(specification, dict) or not isinstance(specification.get("job"), str):
                        raise ValueError(f"invalid build job at index {index}")
                    arguments = specification.get("arguments") or {}
                    if not isinstance(arguments, dict):
                        raise ValueError(f"build job arguments must be an object at index {index}")
                    result = await _run_job_on_app(app, specification["job"], arguments)
                    if not result["success"]:
                        failed_results.append(result)
                        _append_build_error(
                            build_log,
                            f"[sandbox] build job {specification['job']!r} failed: "
                            f"{result.get('error') or 'job failed'}\n"
                            f"{result.get('traceback') or ''}",
                        )
                    item = {"job": specification["job"], "result": result}
                    checkpoint = specification.get("memory_checkpoint")
                    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
                        raise ValueError(f"invalid memory_checkpoint at index {index}")
                    if result["success"] and checkpoint is not None:
                        item["memory_checkpoint"] = _commit_memory_checkpoint(case_root, app_config, checkpoint)
                    results.append(item)
            for result in failed_results:
                for diagnostic in result.get("cleanup_errors", []):
                    _append_build_error(build_log, diagnostic["traceback"])
        finally:
            if logger is not None:
                logger.info("[sandbox] build phase finished")
            get_logger(log_to_console=False, log_to_file=False, force_init=True)
    except BaseException:
        _append_build_error(build_log, traceback.format_exc())
        raise
    return {
        "success": len(results) == len(jobs) and all(item["result"]["success"] for item in results),
        "jobs": results,
    }


def _commit_memory_checkpoint(case_root: Path, app_config: dict[str, Any], message: str) -> dict[str, Any]:
    """Commit one host-declared boundary containing durable file-native memory."""

    workspace_root = Path(app_config["workspace_dir"]).resolve()
    try:
        workspace_root.relative_to(case_root.resolve())
    except ValueError as exc:
        raise ValueError("memory checkpoint path escapes the runtime workspace") from exc
    relative_paths: list[str] = []
    for field in ("daily_dir", "digest_dir", "session_dir"):
        path = (workspace_root / app_config[field]).resolve()
        try:
            relative_path = path.relative_to(workspace_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"memory checkpoint {field} escapes the runtime workspace") from exc
        if not relative_path or relative_path == ".":
            raise ValueError(f"memory checkpoint {field} must not be the workspace root")
        if relative_path not in relative_paths:
            relative_paths.append(relative_path)

    def git(*arguments: str, allowed_exit_codes: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *arguments],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode not in allowed_exit_codes:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
        return result

    git("add", "-A", "--", *relative_paths)
    changed_paths = [
        path
        for path in relative_paths
        if git("diff", "--cached", "--quiet", "--", path, allowed_exit_codes=(0, 1)).returncode == 1
    ]
    commit = [
        "-c",
        "user.name=ReMe Sandbox",
        "-c",
        "user.email=reme-sandbox@localhost",
        "commit",
        "--quiet",
        "--allow-empty",
        "--only",
        "-m",
        message,
    ]
    if changed_paths:
        commit.extend(["--", *changed_paths])
    git(*commit)
    return {
        "commit_sha": git("rev-parse", "HEAD").stdout.strip(),
        "message": message,
        "paths": relative_paths,
    }


async def _run_checkpoint(request: dict[str, Any]) -> dict[str, Any]:
    """Commit host-requested memory history using the build checkpoint rules."""
    message = request.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("memory history commit message must be a non-empty string")
    case_root, app_config = _prepare_runtime(request)
    return {"success": True, "memory_checkpoint": _commit_memory_checkpoint(case_root, app_config, message)}


async def _run_queries(request: dict[str, Any], *, write_summary: bool = True) -> dict[str, Any]:
    """Execute answer requests in one candidate Application with isolated logs.

    ``write_summary=False`` is the append-only primitive used by the validation
    scheduler. It runs exactly one independently leased query without sealing
    the case-level artifact directory with ``summary.json``.
    """
    case_root, app_config = _prepare_runtime(request)
    from reme.reme import ReMe
    from reme.utils import get_logger

    queries = request.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("query request requires a non-empty queries list")
    if not write_summary and len(queries) != 1:
        raise ValueError("single-query request requires exactly one query")
    query_ids = [_validate_query_id(query.get("query_id") if isinstance(query, dict) else None) for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs must be unique")
    queries_root = case_root / "queries"
    if (queries_root / "summary.json").exists() or any((queries_root / query_id).exists() for query_id in query_ids):
        raise FileExistsError("query artifacts already exist for this case")
    queries_root.mkdir(parents=True, exist_ok=True)

    app = ReMe(**app_config)
    results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []
    async with _application_scope(app, failed_results):
        try:
            await app.start()
            for query, query_id in zip(queries, query_ids):
                query_dir = queries_root / query_id
                query_dir.mkdir(parents=False, exist_ok=False)
                log_path = query_dir / "answer.log"
                logger = _start_case_log(log_path)
                logger.info(f"[sandbox] query started query_id={query_id!r}")
                answer_result: dict[str, Any] | None = None
                error: str | None = None
                exception: dict[str, Any] | None = None
                try:
                    question = query.get("question")
                    if not isinstance(question, str):
                        raise ValueError(f"query {query_id!r} requires a string question")
                    answer_arguments = dict(query.get("answer_arguments") or {})
                    answer_arguments["query"] = question
                    answer_result = await _run_job_on_app(
                        app,
                        str(query.get("answer_job") or "agentic_answer"),
                        answer_arguments,
                    )
                    if not answer_result["success"]:
                        error = answer_result.get("error") or "answer job failed"
                        exception = answer_result.get("exception")
                except Exception as exc:  # Keep later queries runnable and preserve this query's log.
                    error = f"{type(exc).__name__}: {exc}"
                    exception = _exception_details(exc)
                    traceback_text = traceback.format_exc()
                else:
                    traceback_text = None
                finally:
                    logger.info(f"[sandbox] query finished query_id={query_id!r}")
                    _clear_query_state(app)
                    get_logger(log_to_console=False, log_to_file=False, force_init=True)

                result = {
                    "query_id": query_id,
                    "question": query.get("question"),
                    "answer": answer_result.get("answer") if answer_result is not None else None,
                    "score": None,
                    "answer_result": answer_result,
                    "judge_result": None,
                    "error": error,
                }
                if exception is not None:
                    result["exception"] = exception
                if traceback_text is not None:
                    result["traceback"] = traceback_text
                _atomic_write_json(query_dir / "result.json", result)
                results.append(result)
                if error is not None:
                    failed_results.append(result)
        finally:
            get_logger(log_to_console=False, log_to_file=False, force_init=True)

    for result in failed_results:
        if result.get("cleanup_errors"):
            _atomic_write_json(queries_root / result["query_id"] / "result.json", result)

    summary = {
        "schema_version": 1,
        "case_id": request.get("case_id"),
        "query_count": len(results),
        "scored_count": 0,
        "mean_score": None,
        "queries": [
            {
                "query_id": result["query_id"],
                "score": result["score"],
                **({"error": result["error"]} if result["error"] is not None else {}),
            }
            for result in results
        ],
    }
    if write_summary:
        _atomic_write_json(queries_root / "summary.json", summary)
    return {"success": all(result["error"] is None for result in results), "summary": summary, "queries": results}


def main() -> int:
    """Read one request file and always emit a structured response file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    response_path = Path(args.response)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        mode = request.get("mode", "job")
        if mode == "job":
            operation = _run(request)
        elif mode == "build":
            operation = _run_build(request)
        elif mode == "queries":
            operation = _run_queries(request)
        elif mode == "query":
            operation = _run_queries(request, write_summary=False)
        elif mode == "checkpoint":
            operation = _run_checkpoint(request)
        else:
            raise ValueError(f"unknown sandbox worker mode: {mode!r}")
        result = asyncio.run(operation)
        exit_code = 0 if result["success"] else 1
    except Exception as exc:  # The host needs an artifact even for broken candidates.
        result = {
            "success": False,
            "answer": "",
            "metadata": {},
            "token_usage": {},
            "job_call_counts": {},
            "error": f"{type(exc).__name__}: {exc}",
            "exception": _exception_details(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    response_path.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
