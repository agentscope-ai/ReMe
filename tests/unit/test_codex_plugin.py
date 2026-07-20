"""Tests for the Codex plugin packaging and structure."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path

PLUGIN_ROOT = Path(__file__).parents[2] / "plugins" / "codex" / "reme"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestPluginJson:
    """Validate .codex-plugin/plugin.json structure."""

    def test_plugin_json_exists(self):
        assert (PLUGIN_ROOT / ".codex-plugin" / "plugin.json").is_file()

    def test_plugin_json_valid(self):
        data = _read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")
        assert data["name"] == "reme"
        assert "version" in data
        assert "description" in data
        assert "skills" in data
        assert "mcpServers" in data
        assert "hooks" in data

    def test_plugin_references_existing_paths(self):
        data = _read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")
        skills_dir = data.get("skills", "")
        mcp_file = data.get("mcpServers", "")
        hooks_file = data.get("hooks", "")
        assert (PLUGIN_ROOT / skills_dir).is_dir(), f"skills dir not found: {skills_dir}"
        assert (PLUGIN_ROOT / mcp_file).is_file(), f"mcp config not found: {mcp_file}"
        assert (PLUGIN_ROOT / hooks_file).is_file(), f"hooks config not found: {hooks_file}"

    def test_plugin_interface_fields(self):
        data = _read_json(PLUGIN_ROOT / ".codex-plugin" / "plugin.json")
        iface = data.get("interface", {})
        assert iface.get("displayName") == "ReMe Memory"
        assert iface.get("category") == "Productivity"


class TestMcpJson:
    """Validate .mcp.json structure and port."""

    def test_mcp_json_valid(self):
        data = _read_json(PLUGIN_ROOT / ".mcp.json")
        servers = data.get("mcpServers", {})
        assert "reme" in servers
        assert servers["reme"]["type"] == "http"

    def test_mcp_url_uses_default_port_2333(self):
        data = _read_json(PLUGIN_ROOT / ".mcp.json")
        url = data["mcpServers"]["reme"]["url"]
        assert ":2333" in url, f"expected port 2333, got {url}"


class TestHooksJson:
    """Validate hooks.json structure."""

    def test_hooks_json_valid(self):
        data = _read_json(PLUGIN_ROOT / "hooks" / "hooks.json")
        hooks = data.get("hooks", {})
        assert "Stop" in hooks

    def test_hook_command_has_windows_path(self):
        data = _read_json(PLUGIN_ROOT / "hooks" / "hooks.json")
        stop_hooks = data["hooks"]["Stop"]
        hook_block = stop_hooks[0]["hooks"][0]
        assert "commandWindows" in hook_block, "hook must declare commandWindows"
        assert hook_block.get("timeout") == 30

    def test_hook_command_points_to_existing_script(self):
        """The hook command path is relative to PLUGIN_ROOT — verify the script exists."""
        assert (PLUGIN_ROOT / "hooks" / "auto_memory.py").is_file()


class TestSkillMarkdown:
    """Validate the reme-memory skill."""

    def test_skill_md_exists_and_has_frontmatter(self):
        skill_path = PLUGIN_ROOT / "skills" / "reme-memory" / "SKILL.md"
        assert skill_path.is_file()
        content = skill_path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "name: reme-memory" in content
        assert "description:" in content

    def test_skill_references_correct_port(self):
        content = (PLUGIN_ROOT / "skills" / "reme-memory" / "SKILL.md").read_text(encoding="utf-8")
        assert "2333" in content, "skill should reference default port 2333"
