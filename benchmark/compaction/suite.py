"""Freeze rule-scored, read-only adaptations of fourteen distinct fixture tasks."""

import ast
import hashlib
import importlib.util
import json
import re
import random
import subprocess
import sys

from benchmark.compaction.cases import GOLD, exact_match, write


def module(path, name):
    """Load fixture definitions only; never run their CLI or agent workers."""
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def context(text, pattern, radius=10):
    """Freeze original matching lines with context, with no generated filler or answer lookup."""
    lines = text.splitlines(keepends=True)
    indices = {
        j
        for i, line in enumerate(lines)
        if re.search(pattern, line)
        for j in range(max(0, i - radius), min(len(lines), i + radius + 1))
    }
    return "".join(f"{i+1}|{lines[i]}" for i in sorted(indices))


def planted_session(path, function, session_id, position):
    """Extract a literal upstream seeded history without running DB setup or worker code."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == function:
            if any(isinstance(a, ast.Constant) and a.value == session_id for a in node.args):
                return ast.literal_eval(node.args[position])
    raise ValueError("Expected upstream seeded session")


def define(task_id, query, files, gold, fields, origin, *, query_only=False, unordered=()):
    """Keep fixture answers outside the read-only tool workspace and compressor inputs."""
    return {
        "id": task_id,
        "query": query,
        "files": files,
        "gold": gold,
        "fields": fields,
        "origin": origin,
        "query_only": query_only,
        "unordered": list(unordered),
    }


def definitions(hermes, scratch):
    """Use pinned upstream fixture bytes; adapt actions to read-only investigation explicitly."""
    core = module(hermes / "evals/core_tool_deferral/tasks.py", "compaction_core_tasks")
    perf = module(hermes / "evals/toolperf_abeval/ab_eval.py", "compaction_perf_tasks")
    read = module(hermes / "evals/readtool/fixtures.py", "compaction_read_fixtures")
    core_root, perf_root, read_root = (scratch / name for name in ("core", "perf", "read"))
    for root in (core_root, perf_root, read_root):
        root.mkdir(parents=True)
    for name in ("longrange_backup_pipeline", "eager_refactor_control", "todo_three_files", "config_grep_distractor"):
        root = core_root / name
        root.mkdir()
        core.TASKS_BY_ID[name]["fixtures"](str(root))
    perf.make_sandbox(perf_root)
    read._write_package_json(read_root)  # pylint: disable=protected-access
    read._write_app_js(read_root)  # pylint: disable=protected-access
    read._write_unicode_notes(read_root)  # pylint: disable=protected-access
    read._write_server_log(read_root, random.Random(20260809))  # pylint: disable=protected-access
    read._write_lockfile(read_root, random.Random(20260809))  # pylint: disable=protected-access
    backup = planted_session(hermes / "evals/core_tool_deferral/worker.py", "sess", "seed_backup_talk", 1)
    incident = planted_session(hermes / "evals/session_search_schema/fixtures.py", "mk", "20260818_140200_cc22dd", 4)

    def core_files(name):
        root = core_root / name
        return {str(p.relative_to(root)): p.read_text() for p in root.rglob("*") if p.is_file()}

    def core_query(name):
        return core.TASKS_BY_ID[name]["prompt"]

    def perf_files(*names):
        return {name: (perf_root / "proj" / name).read_text() for name in names}

    def perf_query(name):
        return perf.TASKS[name].replace("{WORK}/proj", ".")

    rows = []
    rows.append(
        define(
            "payment",
            "Investigate the original payment retry incident.",
            {},
            GOLD,
            {k: k.replace("_", " ") for k in GOLD},
            "Original Hermes payment fixture",
            unordered=("no_retry_http_statuses",),
        ),
    )
    files = core_files("longrange_backup_pipeline")
    files["session_history.json"] = json.dumps(backup, indent=2)
    rows.append(
        define(
            "backup",
            core_query("longrange_backup_pipeline"),
            files,
            {"repo": "sparks-data", "ignored_repo": "toybox", "data_text": "important"},
            {
                "repo": "repository requiring backup",
                "ignored_repo": "repository explicitly excluded",
                "data_text": "contents of the target repository data file",
            },
            "core_tool_deferral/longrange_backup_pipeline",
        ),
    )
    rows.append(
        define(
            "refactor",
            core_query("eager_refactor_control"),
            core_files("eager_refactor_control"),
            {
                "old_timeout": 30,
                "call_sites": 3,
                "functions": ["fetch_a", "fetch_b", "fetch_c"],
                "methods": ["get", "post", "head"],
                "old_version": "1.0",
            },
            {
                "old_timeout": "original timeout integer",
                "call_sites": "number of requests call sites",
                "functions": "function names in source order",
                "methods": "HTTP methods in source order",
                "old_version": "existing changelog version",
            },
            "core_tool_deferral/eager_refactor_control",
        ),
    )
    rows.append(
        define(
            "three_files",
            core_query("todo_three_files"),
            core_files("todo_three_files"),
            {"file_count": 3, "header": "item,qty", "data_rows": 2},
            {"file_count": "required output file count", "header": "CSV header", "data_rows": "CSV data row count"},
            "core_tool_deferral/todo_three_files",
            query_only=True,
        ),
    )
    rows.append(
        define(
            "active_config",
            core_query("config_grep_distractor"),
            core_files("config_grep_distractor"),
            {"retry_limit": 7, "backoff": 2, "active_file": "conf/app.ini", "old_file": "conf/old.ini"},
            {
                "retry_limit": "active integer value",
                "backoff": "active integer backoff",
                "active_file": "workspace-relative active config",
                "old_file": "workspace-relative retired config",
            },
            "core_tool_deferral/config_grep_distractor",
        ),
    )
    rows.append(
        define(
            "checkout_504",
            (
                "Investigate the checkout 504 incident: distinguish the failed rollback, "
                "root cause and final database fix."
            ),
            {"session_history.json": json.dumps(incident, indent=2)},
            {
                "setting": "statement_timeout",
                "value_seconds": 45,
                "index_mode": "CONCURRENTLY",
                "root_cause": "CREATE INDEX without CONCURRENTLY",
                "rolled_back_version": "v2.30.2",
                "rollback_fixed": False,
                "deployed_version": "v2.31.0",
                "initial_error_percent": 8,
            },
            {
                "setting": "final database timeout setting name",
                "value_seconds": "final timeout in seconds",
                "index_mode": "mode used to recreate the index",
                "root_cause": "SQL operation that competed with checkout writes",
                "rolled_back_version": "rollback target version",
                "rollback_fixed": "did rollback resolve the incident, boolean",
                "deployed_version": "version deployed before the incident",
                "initial_error_percent": "initial request error percentage as integer",
            },
            "session_search_schema/t2_scroll",
        ),
    )
    rows.append(
        define(
            "dependency_version",
            (
                "What exact left-pad version is pinned, and which file declares the test script? "
                "Record the package scripts for a handoff."
            ),
            {
                "package.json": (read_root / "package.json").read_text(),
                "package-lock.head.txt": "".join(
                    (read_root / "package-lock.json").read_text().splitlines(keepends=True)[:40],
                ),
            },
            {
                "version": "1.3.0",
                "declaration": "package.json",
                "test": "vitest run",
                "build": "tsc -p .",
                "package_version": "4.2.1",
            },
            {
                "version": "pinned left-pad version",
                "declaration": "file declaring the test script",
                "test": "exact test command",
                "build": "exact build command",
                "package_version": "project package version",
            },
            "readtool/lockfile_version",
        ),
    )
    tree = ast.parse((hermes / "evals/readtool/fixtures.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_write_minified_bundle")
    parts = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "parts" for t in n.targets)
    )
    fragment = ast.literal_eval(parts.value)[0]
    rows.append(
        define(
            "retry_backoff",
            "Find the retry-delay function in src/ and explain its base, growth, cap and jitter.",
            {"src/app.js": (read_root / "src/app.js").read_text(), "src/app.min.head.txt": fragment},
            {
                "base_ms": 250,
                "cap_ms": 30000,
                "growth_factor": 2,
                "jitter": "uniform_integer_below_bound",
                "function": "retryDelay",
            },
            {
                "base_ms": "base delay in milliseconds",
                "cap_ms": "maximum delay in milliseconds",
                "growth_factor": "exponential multiplier per attempt",
                "jitter": "choose none or uniform_integer_below_bound",
                "function": "function name",
            },
            "readtool/minified_backoff",
        ),
    )
    log = (read_root / "logs/server.log").read_text()
    rows.append(
        define(
            "server_error",
            (
                "Locate the unique ERROR in the server log and retain its exact request, timestamp, "
                "upstream and connection failure."
            ),
            {
                "server.head.txt": "".join(log.splitlines(keepends=True)[:20]),
                "server.error-context.txt": context(log, r"\bERROR\b"),
            },
            {
                "request_id": "req-7f3d9",
                "timestamp": "2026-08-08T23:41:17Z",
                "status": 502,
                "upstream": "payments",
                "error": "ECONNREFUSED",
                "address": "10.0.4.17:8443",
            },
            {
                "request_id": "error request ID",
                "timestamp": "full UTC error timestamp",
                "status": "HTTP integer status",
                "upstream": "failed upstream service",
                "error": "connection error code",
                "address": "upstream host and port",
            },
            "readtool/log_error_hunt",
        ),
    )
    notes = next((read_root / "notes").iterdir())
    rows.append(
        define(
            "meeting_notes",
            "Read the meeting notes and report the third bullet exactly, retaining the other agreed actions.",
            {"notes.txt": notes.read_text()},
            {
                "third": "rotate the API keys quarterly",
                "first": "ship the payments retry fix",
                "second": "audit the staging TLS certs",
                "fourth": "close out the Q3 incident review",
            },
            {k: f"exact {k} bullet" for k in ("third", "first", "second", "fourth")},
            "readtool/unicode_filename",
        ),
    )
    rows.append(
        define(
            "targeted_edit",
            perf_query("err_ambiguous_edit"),
            perf_files("handlers.py", "check_handlers.py"),
            {
                "target": "slow_handler",
                "original_timeouts": {"fast_handler": 10, "slow_handler": 10, "medium_handler": 10},
                "required_timeouts": {"fast_handler": 10, "slow_handler": 60, "medium_handler": 10},
                "success_marker": "HANDLERS_OK_552",
            },
            {
                "target": "function requiring modification",
                "original_timeouts": "mapping of all original function timeouts",
                "required_timeouts": "mapping after the requested fix",
                "success_marker": "checker success marker",
            },
            "toolperf_abeval/err_ambiguous_edit",
        ),
    )
    output = subprocess.check_output([sys.executable, str(perf_root / "proj/noisy_build.py")], text=True)
    rows.append(
        define(
            "build_token",
            perf_query("err_big_output"),
            {
                **perf_files("noisy_build.py"),
                "build.head.txt": "".join(output.splitlines(keepends=True)[:20]),
                "build.token-context.txt": context(output, r"^UNIQUE_TOKEN="),
            },
            {"token": "tok_9f31c_middle", "step": 2000, "total_steps": 4000, "prefix": "UNIQUE_TOKEN="},
            {
                "token": "full unique token value without its prefix",
                "step": "build loop step that prints the token",
                "total_steps": "build loop iteration count",
                "prefix": "exact token line prefix",
            },
            "toolperf_abeval/err_big_output",
        ),
    )
    rows.append(
        define(
            "package_versions",
            perf_query("err_multi_dir"),
            perf_files("pkg_a/version.txt", "pkg_b/version.txt", "pkg_c/version.txt"),
            {"versions": {"pkg_a": "1.4.2", "pkg_b": "0.9.7", "pkg_c": "3.2.1"}, "joined": "1.4.2,0.9.7,3.2.1"},
            {
                "versions": "mapping from package directory to exact version",
                "joined": "comma-separated versions in a,b,c order",
            },
            "toolperf_abeval/err_multi_dir",
        ),
    )
    anomaly = (perf_root / "proj/records.log").read_text()
    rows.append(
        define(
            "checksum_anomaly",
            perf_query("err_big_file_read"),
            {
                "records.head.txt": "".join(anomaly.splitlines(keepends=True)[:20]),
                "records.anomaly-context.txt": context(anomaly, r"^ANOMALY:"),
            },
            {
                "line": "ANOMALY: checksum drift detected in shard 7 (code X99Q)",
                "code": "X99Q",
                "shard": 7,
                "line_number": 4201,
            },
            {
                "line": "full anomaly line without display line number",
                "code": "anomaly code",
                "shard": "integer shard number",
                "line_number": "one-based original line number",
            },
            "toolperf_abeval/err_big_file_read",
        ),
    )
    return rows


def exam(case):
    """Supply field names/descriptions, never their expected values, to the answering model."""
    prompt = (
        'Using only recorded evidence, return JSON {"facts": {...}}. No tools. Missing facts must be null. '
        "Use exact strings and JSON types; numbers are integers. Answer these fields: "
        + json.dumps(case["fields"], ensure_ascii=False)
    )
    if case["id"] == "payment":
        prompt += (
            " regression_attempt_counts must be a JSON object with keys "
            "timeout_then_success, validation_422 and three_timeouts, each mapping to an integer count; "
            "do not return an array."
        )
    if case["id"] == "meeting_notes":
        prompt += " Return the bullet text without its leading list marker (-), indentation or surrounding whitespace."
    return prompt


def grade(case, prediction):
    """Type-strict field checks; declared unordered lists ignore order, with no LLM judge."""
    observed = prediction.get("facts", {}) if isinstance(prediction, dict) else {}
    observed = observed if isinstance(observed, dict) else {}
    checks = {}
    for key, expected in case["gold"].items():
        value = observed.get(key)
        if key in case["unordered"] and isinstance(value, list):
            value, expected = sorted(value, key=str), sorted(expected, key=str)
        checks[key] = exact_match(value, expected)
    return checks


def freeze(hermes, output):
    """Write all queries, answers and rules before model calls; runtime artifacts stay outside git."""
    output.mkdir(parents=True)
    rows = definitions(hermes, output / "upstream-fixtures")
    for case in rows:
        root = output / case["id"]
        work = root / "workspace"
        work.mkdir(parents=True)
        for name, text in case["files"].items():
            path = work / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        case["source_chars"] = sum(map(len, case["files"].values()))
        case["control_reason"] = (
            "query_only" if case["query_only"] else "short_fixture" if case["source_chars"] < 2000 else None
        )
        if case["id"] == "payment":
            case["control_reason"] = None
        files = case.pop("files")
        case["source_hashes"] = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()}
        prompt = (
            "Read-only evidence collection for this task: " + case["query"] + "\n"
            "Read ALL of these snapshot files completely using read_file: " + ", ".join(files) + ". "
            "Then report findings and proposed actions only. Do not modify files or run commands. "
            "Snapshots preserve selected original fixture regions; display line numbers refer to the original file."
        )
        write(
            root / "collection-task.json",
            {"prompt": prompt, "model": "qwen3.8-max", "source_hashes": case["source_hashes"]},
        )
    manifest = {
        "hermes_commit": subprocess.check_output(["git", "-C", str(hermes), "rev-parse", "HEAD"], text=True).strip(),
        "source": "Local Hermes MIT-licensed task fixtures; explicitly adapted read-only",
        "cases": rows,
    }
    write(output / "corpus.json", manifest)
    return manifest
