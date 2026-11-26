"""Opt-in native Claude Code validation of both generated MCP connection modes."""

# pylint: disable=missing-function-docstring

import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("transport", ["http", "stdio"])
def test_generated_claude_plugin_validates_without_model_calls(tmp_path, monkeypatch, transport):
    binary = os.environ.get("REME_CLAUDE_BIN")
    if not binary:
        pytest.skip("Set REME_CLAUDE_BIN to run the native Claude Code manifest check")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-home"))
    script = Path(__file__).resolve().parents[2] / "integrations/configure_mcp.py"
    spec = importlib.util.spec_from_file_location("reme_configure_mcp", script)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    marketplace = setup.configure("claude-code", transport)
    for target in (marketplace, marketplace / "reme"):
        result = subprocess.run(
            [binary, "plugin", "validate", str(target)],
            cwd=tmp_path,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
