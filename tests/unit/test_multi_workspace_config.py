"""Multi-workspace workspace_dir configuration contracts (Phase 1)."""

# pylint: disable=protected-access,missing-function-docstring

from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

from reme.application import Application
from reme.components import ApplicationContext
from reme.config import resolve_app_config
from reme.schema import ApplicationConfig


def test_string_workspace_normalizes_to_default_alias(tmp_path):
    config = ApplicationConfig(workspace_dir=str(tmp_path / "ws"))

    assert config.workspace_dir == str(tmp_path / "ws")
    assert config.workspaces == {"default": str(tmp_path / "ws")}
    assert "workspaces" not in config.model_dump(mode="json")
    assert "workspaces" not in config.model_dump()


def test_mapping_workspace_keeps_order_and_first_entry_as_default(tmp_path):
    config = ApplicationConfig(
        workspace_dir={
            "beta": str(tmp_path / "b"),
            "alpha": str(tmp_path / "a"),
        },
    )

    assert list(config.workspaces) == ["beta", "alpha"]
    assert config.workspace_dir == str(tmp_path / "b")
    assert config.workspaces["beta"] == config.workspace_dir


def test_mapping_workspace_expands_home_and_dedupes_by_resolved_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = ApplicationConfig(workspace_dir={"home": "~/ws", "other": str(tmp_path / "ws2")})

    assert config.workspaces["home"] == str(tmp_path / "ws")
    with pytest.raises(ValidationError, match="distinct directories"):
        ApplicationConfig(workspace_dir={"one": "~/ws", "two": str(tmp_path / "ws")})


def test_empty_mapping_rejected():
    with pytest.raises(ValidationError, match="at least one workspace"):
        ApplicationConfig(workspace_dir={})


def test_empty_alias_rejected(tmp_path):
    with pytest.raises(ValidationError, match="aliases must be non-empty"):
        ApplicationConfig(workspace_dir={"  ": str(tmp_path / "ws")})


def test_overlapping_directories_rejected(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)

    with pytest.raises(ValidationError, match="must not overlap"):
        ApplicationConfig(workspace_dir={"parent": str(parent), "child": str(child)})
    with pytest.raises(ValidationError, match="must not overlap"):
        ApplicationConfig(workspace_dir={"child": str(child), "parent": str(parent)})

    sibling = tmp_path / "parent-sibling"
    config = ApplicationConfig(workspace_dir={"parent": str(parent), "sibling": str(sibling)})
    assert list(config.workspaces) == ["parent", "sibling"]


def test_application_context_exposes_ordered_workspace_contexts(tmp_path):
    context = ApplicationContext(
        workspace_dir={"main": str(tmp_path / "main"), "scratch": str(tmp_path / "scratch")},
    )

    assert list(context.workspaces) == ["main", "scratch"]
    assert context.workspace is context.workspaces["main"]
    main = context.workspaces["main"]
    assert main.alias == "main"
    assert main.workspace_path == tmp_path / "main"
    assert main.metadata_path == tmp_path / "main" / "metadata"
    assert main.session_path == tmp_path / "main" / "session"
    assert main.mem_session_path == tmp_path / "main" / "mem_session"
    assert main.resource_path == tmp_path / "main" / "resource"
    assert main.daily_path == tmp_path / "main" / "daily"
    assert main.digest_path == tmp_path / "main" / "digest"


def test_workspace_context_skips_empty_subdirs(tmp_path):
    context = ApplicationContext(workspace_dir=str(tmp_path / "ws"), digest_dir="")
    workspace = context.workspace

    assert workspace.digest_path is None
    assert tmp_path / "ws" / "digest" not in workspace.storage_dirs()
    assert len(workspace.storage_dirs()) == 5


def test_string_workspace_layout_unchanged(tmp_path):
    app = Application(
        workspace_dir=str(tmp_path / "ws"),
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )

    root = tmp_path / "ws"
    assert root.is_dir()
    for subdir in ("metadata", "session", "mem_session", "resource", "daily", "digest"):
        assert (root / subdir).is_dir()
    assert list(app.context.workspaces) == ["default"]
    assert app.context.workspace.workspace_path == root


def test_application_creates_directories_for_every_alias(tmp_path):
    Application(
        workspace_dir={"main": str(tmp_path / "main"), "scratch": str(tmp_path / "scratch")},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )

    for alias in ("main", "scratch"):
        root = tmp_path / alias
        assert root.is_dir()
        for subdir in ("metadata", "session", "mem_session", "resource", "daily", "digest"):
            assert (root / subdir).is_dir()


def test_yaml_config_loads_mapping_workspace(tmp_path):
    config_path = tmp_path / "multi.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "workspace_dir": {"main": str(tmp_path / "m"), "scratch": str(tmp_path / "s")},
                "enable_logo": False,
            },
        ),
        encoding="utf-8",
    )

    resolved = resolve_app_config(config=str(config_path), log_config=False)
    config = ApplicationConfig(**resolved)

    assert list(config.workspaces) == ["main", "scratch"]
    assert config.workspace_dir == str(tmp_path / "m")


def test_default_yaml_string_workspace_still_loads(monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    resolved = resolve_app_config(log_config=False)
    config = ApplicationConfig(**resolved)

    assert isinstance(config.workspace_dir, str)
    assert list(config.workspaces) == ["default"]
    assert config.workspaces["default"] == str(Path(".reme").resolve())


def test_alias_normalization_collision_rejected(tmp_path):
    with pytest.raises(ValidationError, match="aliases collide"):
        ApplicationConfig(workspace_dir={"alice": str(tmp_path / "a"), " alice ": str(tmp_path / "b")})


def test_non_string_alias_rejected(tmp_path):
    with pytest.raises(ValidationError, match="aliases must be strings"):
        ApplicationConfig(workspace_dir={123: str(tmp_path / "a")})


@pytest.mark.parametrize("bad_value", [None, 123, "", "  "])
def test_invalid_mapping_value_rejected(bad_value):
    with pytest.raises(ValidationError, match="non-empty path string"):
        ApplicationConfig(workspace_dir={"alice": bad_value})


def test_supplied_workspaces_rejected_with_string_form(tmp_path):
    with pytest.raises(ValidationError, match="derived from workspace_dir"):
        ApplicationConfig(workspace_dir=str(tmp_path / "expected"), workspaces={"": str(tmp_path / "other")})


def test_supplied_workspaces_rejected_with_mapping_form(tmp_path):
    with pytest.raises(ValidationError, match="derived from workspace_dir"):
        ApplicationConfig(
            workspace_dir={"main": str(tmp_path / "main")},
            workspaces={"other": str(tmp_path / "other")},
        )


def test_default_workspace_matches_directory_creation_source(tmp_path):
    """The default workspace_dir and the materialized directories share one derivation."""
    app = Application(
        workspace_dir={"main": str(tmp_path / "main"), "scratch": str(tmp_path / "scratch")},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
    )

    assert app.config.workspace_dir == str(tmp_path / "main")
    assert app.config.workspace_dir == app.config.workspaces["main"]
    assert app.context.workspace.workspace_path == tmp_path / "main"
    assert (tmp_path / "main" / "daily").is_dir()
