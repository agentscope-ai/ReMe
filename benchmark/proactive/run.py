"""Opt-in provider benchmark for the current chat, task and proactive agents."""

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path

from benchmark.proactive.cases import CASES
from benchmark.proactive.evaluate import grade, report
from benchmark.proactive.prompts import freeze
from benchmark.proactive.runtime import EFFORT, execute, save
from reme.config import resolve_app_config
from reme.utils.env_utils import parse_env_file

REPO = Path(__file__).resolve().parents[2]


def digest(value) -> str:
    """Hash JSON inputs deterministically for reproducible manifests."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def sources() -> dict:
    """Fingerprint the source files that define this benchmark."""
    paths = list((REPO / "reme/steps/proactive").rglob("*.py")) + list((REPO / "reme/steps/proactive").glob("*.yaml"))
    paths += list((REPO / "reme/components").glob("*pa*.py"))
    paths += [REPO / "reme/config/proactive.yaml", REPO / "reme/components/agent_wrapper/codex_agent_wrapper.py"]
    paths += [REPO / "reme/components/codex_app_server.py", REPO / "reme/schema/reminder.py"]
    paths += list((REPO / "reme/schema").glob("pa*.py"))
    paths += list((REPO / "benchmark/proactive").glob("*.py"))
    return {path.relative_to(REPO).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def credentials() -> dict:
    """Read explicit provider credentials without persisting them."""
    path = REPO / ".env"
    values = parse_env_file(path) if path.exists() else {}
    auth = {key: os.environ.get(key) or values.get(key) for key in ("LLM_API_KEY", "LLM_BASE_URL")}
    if not all(auth.values()):
        raise ValueError("LLM_API_KEY and LLM_BASE_URL are required")
    return auth


def initialize(root: Path, *, models=None, repeats: int | None = None) -> dict:
    """Create or validate a frozen benchmark manifest."""
    if root.resolve().is_relative_to(REPO):
        raise ValueError("Benchmark output must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    inputs = {"prompts": freeze(REPO), "cases": CASES, "source_hashes": sources()}
    fingerprint = digest(inputs)
    path = root / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Source or inputs changed; choose a new output directory")
        if models is not None and models != manifest["models"]:
            raise ValueError("Model selection changed; choose a new output directory")
        if repeats is not None and repeats != manifest["profile"]["repeats"]:
            raise ValueError("Repeat count changed; choose a new output directory")
        return manifest
    cfg = resolve_app_config(config="proactive", log_config=False)
    manifest = {
        **inputs,
        "fingerprint": fingerprint,
        "seed": 20261108,
        "models": models or [cfg["components"]["agent_wrapper"]["default"]["model"]],
        "profile": {"effort": EFFORT, "repeats": repeats or 1, "clock": "fixed", "evidence": "synthetic"},
    }
    save(path, manifest)
    return manifest


def make_tasks(manifest: dict) -> list[dict]:
    """Create a deterministic matrix with safe, unique output paths."""
    model_keys = {
        model: re.sub(r"[^A-Za-z0-9_-]", "_", model)[:64] + "-" + digest(model)[:12] for model in manifest["models"]
    }
    tasks = [
        {
            "id": f"{model_keys[model]}__{prompt}__{case}__{repeat}",
            "model": model,
            "prompt": prompt,
            "case": case,
            "repeat": repeat,
            "role": fixture["role"],
        }
        for model in manifest["models"]
        for prompt in manifest["prompts"]
        for case, fixture in manifest["cases"].items()
        for repeat in range(manifest["profile"]["repeats"])
    ]
    random.Random(manifest["seed"]).shuffle(tasks)
    return tasks


async def run(args) -> None:
    """Execute unfinished matrix entries and save their results."""
    manifest = initialize(args.root, models=args.models.split(",") if args.models else None, repeats=args.repeats)
    auth = credentials()
    cfg = resolve_app_config(config="proactive", log_config=False)
    cfg["components"]["codex_app_server"]["default"].update(api_key=auth["LLM_API_KEY"], base_url=auth["LLM_BASE_URL"])
    tasks = make_tasks(manifest)
    if args.cases:
        selected = set(args.cases.split(","))
        if selected - manifest["cases"].keys():
            raise ValueError("Unknown case name")
        tasks = [task for task in tasks if task["case"] in selected]
    for task in tasks:
        folder = args.root / "cases" / task["id"]
        if (folder / "result.json").exists():
            continue
        record = await execute(
            cfg,
            task["model"],
            manifest["prompts"][task["prompt"]],
            manifest["cases"][task["case"]],
            folder,
            args.timeout,
        )
        result = {**task, "record": record, "grade": grade(task["case"], record)}
        save(folder / "result.json", result)
        print(
            json.dumps({"id": task["id"], "score": result["grade"]["score"], "completed": record["completed"]}),
            flush=True,
        )
    save(args.root / "report.json", report(args.root))


def main():
    """Parse arguments for the opt-in provider benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", help="Comma-separated model IDs; defaults to the configured model")
    parser.add_argument("--cases", help="Comma-separated case names")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (args.repeats is not None and args.repeats < 1) or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    args.root = args.root.expanduser().resolve()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
