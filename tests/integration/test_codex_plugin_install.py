"""Opt-in native plugin discovery regression, without model calls or hook execution.

Run with REME_CODEX_BIN=/path/to/codex pytest tests/integration/test_codex_plugin_install.py.
The selected runtime must support the hooks/list app-server API.
"""

# pylint: disable=missing-function-docstring

import json
import os
import queue
import subprocess
import threading
from pathlib import Path

import pytest

MARKETPLACE = Path(__file__).resolve().parents[2] / "integrations/codex"
EVENTS = {"sessionStart", "sessionEnd", "userPromptSubmit", "stop"}


def test_installed_plugin_exposes_all_hooks_before_trust(tmp_path):
    binary = os.environ.get("REME_CODEX_BIN")
    if not binary:
        pytest.skip("Set REME_CODEX_BIN to run the native Codex installation check")
    home = tmp_path / "codex-home"
    home.mkdir()
    env = {**os.environ, "CODEX_HOME": str(home)}
    for args in (
        ["plugin", "marketplace", "add", str(MARKETPLACE)],
        ["plugin", "add", "reme@reme-codex"],
    ):
        result = subprocess.run([binary, *args], env=env, capture_output=True, text=True, check=False, timeout=60)
        assert result.returncode == 0, result.stderr

    with (
        (tmp_path / "app-server.log").open("w", encoding="utf-8") as stderr,
        subprocess.Popen(
            [binary, "app-server"],
            env=env,
            cwd=tmp_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        ) as process,
    ):
        responses = queue.Queue()

        def receive():
            for line in process.stdout:
                responses.put(json.loads(line))

        reader = threading.Thread(target=receive, daemon=True)
        reader.start()

        def request(request_id, method, params):
            process.stdin.write(json.dumps({"id": request_id, "method": method, "params": params}) + "\n")
            process.stdin.flush()
            while True:
                response = responses.get(timeout=30)
                if response.get("id") == request_id:
                    assert "error" not in response, response
                    return response["result"]

        try:
            request(
                0,
                "initialize",
                {
                    "clientInfo": {"name": "reme-install-test", "version": "1.0"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            process.stdin.write(json.dumps({"method": "initialized", "params": {}}) + "\n")
            process.stdin.flush()
            details = request(
                1,
                "plugin/read",
                {"marketplacePath": str(MARKETPLACE / ".agents/plugins/marketplace.json"), "pluginName": "reme"},
            )["plugin"]
            assert details["summary"]["installed"] and details["summary"]["enabled"]
            assert details["mcpServers"] == ["reme"]
            assert {hook["eventName"] for hook in details["hooks"]} == EVENTS

            entries = request(2, "hooks/list", {"cwds": [str(tmp_path)]})["data"]
            assert len(entries) == 1
            assert not entries[0]["errors"]
            hooks = [hook for hook in entries[0]["hooks"] if hook.get("pluginId") == "reme@reme-codex"]
            assert len(hooks) == 4
            assert {hook["eventName"] for hook in hooks} == EVENTS
            assert all(hook["enabled"] and hook["trustStatus"] == "untrusted" for hook in hooks)
        finally:
            process.stdin.close()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=5)
            process.stdout.close()
