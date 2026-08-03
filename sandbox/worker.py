"""In-container worker that calls ReMe jobs directly, without HTTP.

The host uploads this file into every case sandbox. It deliberately imports
ReMe only after process startup so source candidates can control exactly which
ReMe package is loaded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
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
        configured = str(getattr(app_config, field)).strip()
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


async def _run(request: dict[str, Any]) -> dict[str, Any]:
    """Construct an Application, run one job, and close it deterministically."""
    job = request["job"]
    job_args = request.get("arguments") or {}
    config = request.get("config") or "lme.yaml"
    workspace_dir = request["workspace_dir"]
    case_root = Path(request["case_root"])
    case_tmp = case_root / "tmp"
    case_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(case_tmp)
    os.chdir(case_root)

    # Import after setting TMPDIR so ReMe and its subprocesses cannot leave
    # case-specific temporary files outside the disposable case root.
    from reme.config import resolve_app_config
    from reme.reme import ReMe
    from reme.schema import ApplicationConfig

    app_config = resolve_app_config(
        config=config,
        workspace_dir=workspace_dir,
        log_to_console=False,
        log_to_file=True,
    )
    app_config["environment"] = dict(os.environ)
    resolved_config = ApplicationConfig.model_validate(app_config)
    _write_runtime_layout(case_root, resolved_config)
    app = ReMe(**app_config)
    try:
        await app.start()
        response = await app.run_job(job, **job_args)
        return {
            "success": bool(response.success),
            "answer": response.answer,
            "metadata": response.metadata,
            "error": None if response.success else str(response.answer),
        }
    finally:
        await app.close()


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
        result = asyncio.run(_run(request))
        exit_code = 0 if result["success"] else 1
    except Exception as exc:  # The host needs an artifact even for broken candidates.
        result = {
            "success": False,
            "answer": "",
            "metadata": {},
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    response_path.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
