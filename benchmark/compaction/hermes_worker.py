"""Collect one genuine Hermes trajectory in an isolated process and read-only fixture workspace."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time

from dotenv import dotenv_values

from benchmark.compaction.cases import write


def collect(root, hermes, env_file):
    """Configure isolated Hermes state before importing its runtime; never save credentials."""
    values = dotenv_values(env_file)
    secrets = [values[k] for k in ("LLM_API_KEY", "DECISION_MODEL_API_KEY") if values.get(k)]
    task = json.loads((root / "collection-task.json").read_text())
    work, home = root / "workspace", root / "hermes_home"
    home.mkdir()
    os.environ.update(
        {
            "HERMES_HOME": str(home),
            "TERMINAL_ENV": "local",
            "TERMINAL_CWD": str(work),
            "TOKENIZERS_PARALLELISM": "false",
        },
    )
    (home / "config.yaml").write_text(
        "compression:\n  enabled: false\n  micro_compact_enabled: false\n"
        f"terminal:\n  backend: local\n  cwd: {work}\n"
        "model:\n  provider: custom\n  context_length: 131072\n",
    )
    sys.path.insert(0, str(hermes))
    from run_agent import AIAgent
    from hermes_state import SessionDB

    requests = []

    def append(name, data):
        text = json.dumps(data, ensure_ascii=False)
        if any(secret in text for secret in secrets):
            raise ValueError("Credential in captured body")
        with (root / name).open("a", encoding="utf-8") as target:
            target.write(text + "\n")

    seen = set()
    for name in ("httpx", "httpx2"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if id(module.Client) in seen:
            continue
        seen.add(id(module.Client))
        original = module.Client.send

        def wrap_send(send):
            def capture(client, request, *args, **kwargs):
                captured = "/chat/completions" in request.url.path
                if captured:
                    body = json.loads(request.content)
                    requests.append(body)
                    append("api_requests.jsonl", body)
                response = send(client, request, *args, **kwargs)
                if captured:
                    response.read()
                    try:
                        body = response.json()
                    except ValueError:
                        body = response.text
                    append("api_responses.jsonl", {"status": response.status_code, "body": body})
                return response

            return capture

        module.Client.send = wrap_send(original)
    agent = None
    database = SessionDB(home / "state.db")
    started = time.monotonic()
    try:
        agent = AIAgent(
            base_url=values["LLM_BASE_URL"],
            api_key=values["LLM_API_KEY"],
            provider="custom",
            api_mode="chat_completions",
            model=task["model"],
            max_iterations=12,
            run_budget_seconds=240,
            max_tokens=1024,
            request_overrides={"reasoning_effort": "none", "extra_body": {"enable_thinking": False}},
            enabled_toolsets=["file"],
            save_trajectories=False,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            skip_background_review=True,
            session_db=database,
            cwd=str(work),
            platform="batch",
        )
        from tools.registry import registry

        entry = registry.get_entry("read_file")
        original_read = entry.handler

        def confined_read(args, **kwargs):
            path = Path(args.get("path", ""))
            resolved = (path if path.is_absolute() else work / path).resolve()
            if not resolved.is_relative_to(work) or str(resolved.relative_to(work)) not in task["source_hashes"]:
                return json.dumps({"error": "Only the listed source snapshots may be read"})
            return original_read(args, **kwargs)

        entry.handler = confined_read
        agent.tools = [t for t in agent.tools if t["function"]["name"] == "read_file"]
        agent.valid_tool_names = {"read_file"}
        agent.context_compressor.enabled = False
        result = agent.run_conversation(
            task["prompt"],
            task_id=root.name,
            system_message=f"Investigate the supplied fixture evidence and propose actions only. "
            f"Use only read_file within {work}. "
            "Do not read environment files, credentials or any path outside this workspace.",
        )
        if not result.get("completed") or not requests:
            raise ValueError("Hermes collection did not complete")
        systems = [m for m in requests[0]["messages"] if m["role"] == "system"]
        history = [*systems, *result["messages"]]
        if len(systems) != 1 or history[-1]["role"] != "assistant" or history[-1].get("tool_calls"):
            raise ValueError("Incomplete system prompt or final assistant report")
        calls = {c["id"]: c for m in history for c in m.get("tool_calls", []) or []}
        results = [m for m in history if m["role"] == "tool"]
        if set(calls) != {m["tool_call_id"] for m in results}:
            raise ValueError("Unpaired tool calls")
        observed = set()
        for message in results:
            call = calls[message["tool_call_id"]]
            args = json.loads(call["function"]["arguments"])
            path = Path(args.get("path") or args.get("file_path"))
            path = (path if path.is_absolute() else work / path).resolve()
            output = json.loads(message["content"])
            if not path.is_relative_to(work) or str(path.relative_to(work)) not in task["source_hashes"]:
                if "error" not in output:
                    raise ValueError("Unexpected content from outside the source allowlist")
                continue  # Denied reads remain in the trajectory; they never reached the filesystem.
            text = "\n".join(re.sub(r"^\d+\|", "", line) for line in output.get("content", "").splitlines())
            if text.rstrip() == path.read_text(encoding="utf-8").rstrip():
                observed.add(str(path.relative_to(work)))
        if observed != set(task["source_hashes"]):
            raise ValueError("Not every source snapshot was read without truncation")
        if any(
            hashlib.sha256((work / name).read_bytes()).hexdigest() != sha for name, sha in task["source_hashes"].items()
        ):
            raise ValueError("Source fixture changed")
        write(root / "trajectory.full.json", {"messages": history})
        write(root / "result.json", result)
        write(root / "session_export.json", database.export_session(agent.session_id, include_compacted=True))
        write(
            root / "collection.json",
            {
                "completed": True,
                "session_id": agent.session_id,
                "messages": len(history),
                "tool_calls": len(calls),
                "articles_read": len(observed),
                "source_unchanged": True,
                "seconds": time.monotonic() - started,
                "trajectory_sha256": hashlib.sha256((root / "trajectory.full.json").read_bytes()).hexdigest(),
            },
        )
        if any(
            secret.encode() in file.read_bytes() for file in root.rglob("*") if file.is_file() for secret in secrets
        ):
            raise ValueError("Credential in artifacts")
        print(root.name, "collected", len(calls), "tool calls", flush=True)
    finally:
        if agent is not None:
            agent.close()
        database.close()


def main():
    """Internal subprocess entry point; provider authorization belongs to the parent collector."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hermes", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()
    collect(args.root.resolve(), args.hermes.resolve(), args.env_file.resolve())


if __name__ == "__main__":
    main()
