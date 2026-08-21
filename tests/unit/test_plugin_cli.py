"""Tests for local plugin package management commands."""

# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

from pathlib import Path
from types import SimpleNamespace

from reme import plugin_cli as plugin_cli_module
from reme import reme as reme_module


class _FakeDistribution:
    def __init__(self, root: Path, name: str = "reme-auto-fin", version: str = "0.1.0") -> None:
        self.root = root
        self.metadata = {"Name": name}
        self.version = version

    def locate_file(self, relative: Path) -> Path:
        return self.root / relative


class _FakeEntryPoint:
    def __init__(self, name: str, value: str, distribution: _FakeDistribution) -> None:
        self.name = name
        self.value = value
        self.dist = distribution


class _FakeEntryPoints(list):
    def select(self, *, group):
        assert group == "reme.plugins"
        return self


def _install_manifest(tmp_path: Path) -> _FakeEntryPoint:
    package = tmp_path / "reme_auto_fin"
    package.mkdir()
    (package / "plugin.yaml").write_text(
        "backends:\n"
        "  auto_fin_step: reme_auto_fin.step:AutoFinStep\n"
        "application_defaults:\n"
        "  jobs:\n"
        "    auto_fin:\n"
        "      backend: base\n",
        encoding="utf-8",
    )
    return _FakeEntryPoint("auto-fin", "reme_auto_fin", _FakeDistribution(tmp_path))


def test_list_plugins_does_not_load_plugin_code(monkeypatch, tmp_path, capsys):
    entry = _install_manifest(tmp_path)
    monkeypatch.setattr(
        plugin_cli_module.metadata,
        "entry_points",
        lambda: _FakeEntryPoints([entry]),
    )

    assert plugin_cli_module.plugin_cli(["list"]) == 0

    output = capsys.readouterr().out
    assert "auto-fin" in output
    assert "reme-auto-fin" in output
    assert "manifest" in output


def test_list_plugins_marks_configured_plugins(monkeypatch, tmp_path, capsys):
    entry = _install_manifest(tmp_path)
    monkeypatch.setattr(
        plugin_cli_module.metadata,
        "entry_points",
        lambda: _FakeEntryPoints([entry]),
    )
    monkeypatch.setattr(plugin_cli_module, "_enabled_plugins", lambda _config: {"auto-fin"})

    assert plugin_cli_module.plugin_cli(["list", "--config", "daily_cookbook"]) == 0

    output = capsys.readouterr().out
    assert "ENABLED" in output
    assert "yes" in output


def test_show_plugin_reads_manifest_without_importing_backends(monkeypatch, tmp_path, capsys):
    entry = _install_manifest(tmp_path)
    monkeypatch.setattr(
        plugin_cli_module.metadata,
        "entry_points",
        lambda: _FakeEntryPoints([entry]),
    )

    assert plugin_cli_module.plugin_cli(["show", "auto-fin"]) == 0

    output = capsys.readouterr().out
    assert "auto_fin_step" in output
    assert "auto_fin" in output


def test_install_uses_current_python_pip(monkeypatch, capsys):
    commands = []
    monkeypatch.setattr(plugin_cli_module, "_run_pip", lambda command: commands.append(command) or 0)

    assert plugin_cli_module.plugin_cli(["install", ".", "--editable", "--upgrade"]) == 0

    assert commands == [["install", "--editable", "--upgrade", "."]]
    assert "plugins list" in capsys.readouterr().out


def test_pip_uses_current_python_interpreter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        plugin_cli_module.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=7),
    )

    assert plugin_cli_module._run_pip(["install", "example"]) == 7

    assert calls == [
        (
            [plugin_cli_module.sys.executable, "-m", "pip", "install", "example"],
            {"check": False},
        ),
    ]


def test_uninstall_resolves_plugin_to_distribution(monkeypatch, tmp_path, capsys):
    entry = _install_manifest(tmp_path)
    commands = []
    monkeypatch.setattr(
        plugin_cli_module.metadata,
        "entry_points",
        lambda: _FakeEntryPoints([entry]),
    )
    monkeypatch.setattr(plugin_cli_module, "_run_pip", lambda command: commands.append(command) or 0)

    assert plugin_cli_module.plugin_cli(["uninstall", "auto-fin", "--yes"]) == 0

    assert commands == [["uninstall", "--yes", "reme-auto-fin"]]
    assert "Remove 'auto-fin'" in capsys.readouterr().out


def test_validate_local_auto_fin_project():
    repository = Path(__file__).resolve().parents[2]

    names = plugin_cli_module._validate_local(repository / "plugins" / "auto-fin")

    assert names == ["auto-fin"]


def test_plugin_command_errors_are_clean(monkeypatch, capsys):
    monkeypatch.setattr(plugin_cli_module.metadata, "entry_points", _FakeEntryPoints)

    assert plugin_cli_module.plugin_cli(["show", "missing"]) == 1

    assert "Plugin 'missing' is not installed" in capsys.readouterr().err


def test_main_routes_plugins_before_loading_environment(monkeypatch):
    events = []
    monkeypatch.setattr("sys.argv", ["reme", "plugins", "list"])
    monkeypatch.setattr(reme_module, "load_env", lambda: events.append("load_env"))
    monkeypatch.setattr(plugin_cli_module, "plugin_cli", lambda argv: events.append(list(argv)) or 0)

    reme_module.main()

    assert events == [["list"]]
