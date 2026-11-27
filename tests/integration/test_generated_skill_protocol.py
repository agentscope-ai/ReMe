"""Packaged Codex skill protocol against a local Responses stub, without credentials."""

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys
from threading import Thread

import pytest

from reme.components.agent_wrapper import BaseAgentWrapper
from reme.core.context import ApplicationContext


@pytest.mark.asyncio
async def test_generated_skill_recovers_after_owner_process_exits(tmp_path):
    """OS locks release after a crash; the next run replaces only its abandoned link and files."""
    script = """
import os
from pathlib import Path
import sys
from reme.components.agent_wrapper.skills import SkillScope
from reme.schema import SkillSpec

root = Path(sys.argv[1]).resolve()
scope = SkillScope([SkillSpec(name="run-task", description="Task", content="Old task")],
                   root / "skills", root / "metadata/agent_skills")
scope.prepare()
scope.publish(root / ".agents/skills")
os._exit(0)
"""
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", script, str(tmp_path))
    try:
        async with asyncio.timeout(20):
            assert await process.wait() == 0
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    target = tmp_path / ".agents/skills/run-task"
    previous = target.resolve()
    assert previous.is_dir()
    wrapper = BaseAgentWrapper(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    definition = {"name": "run-task", "description": "Task", "content": "New task"}
    async with wrapper.skill_scope(skills=[definition]) as options:
        options["_skills"].publish(tmp_path / ".agents/skills")
        current = target.resolve()
        assert current != previous and not previous.exists()
        assert "New task" in (current / "SKILL.md").read_text()
    assert not current.exists() and not target.is_symlink()


def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _texts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _texts(child)


@pytest.mark.asyncio
async def test_generated_skill_uses_native_context_and_survives_follow_up(tmp_path):
    """Explicit selection loads the body once even with automatic skill instructions disabled."""
    from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
    from openai_codex.generated.v2_all import SkillsListResponse

    requests = []

    class LocalResponses(BaseHTTPRequestHandler):
        """Return deterministic model events and retain requests in memory."""

        def log_message(self, *_args):
            pass

        def do_POST(self):  # pylint: disable=invalid-name
            """Serve one deterministic Responses SSE stream."""
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            index = len(requests)
            item = {
                "id": f"msg-{index}",
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done"}],
            }
            response = {
                "id": f"resp-{index}",
                "status": "completed",
                "output": [item],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "total_tokens": 11,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            }
            events = [
                ("response.created", {"response": {"id": f"resp-{index}"}}),
                ("response.output_item.done", {"output_index": 0, "item": item}),
                ("response.completed", {"response": response}),
            ]
            data = "".join(
                f"event: {kind}\ndata: {json.dumps({'type': kind, **payload})}\n\n" for kind, payload in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalResponses)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    root = tmp_path.resolve()
    (root / ".codex").mkdir()
    wrapper = BaseAgentWrapper(app_context=ApplicationContext(workspace_dir=str(root)))
    definition = {
        "name": "run-task",
        "description": "Run the task",
        "content": "Read this source record: native-probe.",
    }
    provider = "model_providers.skill_probe"
    config = CodexConfig(
        cwd=str(root),
        config_overrides=(
            'model_provider="skill_probe"',
            'model="probe"',
            f'{provider}.name="Local skill probe"',
            f'{provider}.base_url="http://127.0.0.1:{server.server_port}/v1"',
            f'{provider}.wire_api="responses"',
            f"{provider}.requires_openai_auth=false",
            f"{provider}.supports_websockets=false",
            "skills.bundled.enabled=false",
            'web_search="disabled"',
        ),
        env={
            "CODEX_HOME": str(root / ".codex"),
            "OPENAI_API_KEY": "",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        },
    )
    try:
        async with asyncio.timeout(45), AsyncCodex(config) as codex:
            async with wrapper.skill_scope(skills=[definition]) as options:
                options["_skills"].publish(root / ".agents/skills")
                discovered = await codex._client.request(  # pylint: disable=protected-access
                    "skills/list",
                    {"cwds": [str(root)], "forceReload": True},
                    response_model=SkillsListResponse,
                )
                assert any(skill.name == "run-task" for entry in discovered.data for skill in entry.skills)
                thread = await codex.thread_start(
                    cwd=str(root),
                    model="probe",
                    sandbox=Sandbox.workspace_write,
                    approval_mode=ApprovalMode.deny_all,
                    config={"skills.include_instructions": False},
                )
                await thread.run("Use $run-task to complete this run.")
                await thread.run("Continue the same task.")
                assert len(requests) == 2
                for request in requests:
                    assert "\n".join(_texts(request)).count(definition["content"]) == 1
            removed = await codex._client.request(  # pylint: disable=protected-access
                "skills/list",
                {"cwds": [str(root)], "forceReload": True},
                response_model=SkillsListResponse,
            )
            assert not any(skill.name == "run-task" for entry in removed.data for skill in entry.skills)
    finally:
        await asyncio.to_thread(server.shutdown)
        server.server_close()
        worker.join(timeout=5)
