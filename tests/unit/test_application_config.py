"""Application configuration contracts."""

# pylint: disable=protected-access

from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfoNotFoundError

import pytest
from agentscope.message import Msg
from pydantic import ValidationError

from reme import Application
from reme.core.context import ApplicationContext
from reme.schema import ApplicationConfig, FileChunk
from reme.steps.evolve.auto_memory import AutoMemoryStep
from reme.steps.file_io.daily_write import DailyWriteStep
from reme.steps.index._source_format import is_session_path, render_chunk_body
from reme.utils.files.file_watch import build_watch_rules
from reme.utils import timezone_utils


@pytest.mark.parametrize(
    "system_timezone,expected",
    [
        ("America/New_York", "America/New_York"),
        (None, "Asia/Shanghai"),
        ("local", "Asia/Shanghai"),
        ("invalid/zone", "Asia/Shanghai"),
    ],
)
def test_application_timezone_defaults_to_system_timezone(monkeypatch, system_timezone, expected):
    """Use the detected IANA zone or fall back to Shanghai."""
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(timezone_utils, "get_localzone_name", Mock(return_value=system_timezone))

    assert ApplicationConfig().timezone == expected


def test_application_timezone_falls_back_when_system_detection_fails(monkeypatch):
    """Unavailable system timezone settings still yield a usable default."""
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(timezone_utils, "get_localzone_name", Mock(side_effect=ZoneInfoNotFoundError))

    assert ApplicationConfig().timezone == "Asia/Shanghai"


def test_application_default_ignores_library_utc_fallback(monkeypatch):
    """A library-supplied UTC fallback is distinct from a detected host name."""
    from zoneinfo import ZoneInfo

    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setattr(timezone_utils, "get_localzone_name", Mock(return_value=None))
    fallback = Mock(return_value=ZoneInfo("Etc/UTC"))
    monkeypatch.setattr(timezone_utils, "get_localzone", fallback)

    assert ApplicationConfig().timezone == "Asia/Shanghai"
    fallback.assert_not_called()


def test_application_timezone_reads_current_tz_environment(monkeypatch):
    """Valid names bypass library detection, even when TZ changes within the process."""
    detect = Mock(return_value="Asia/Shanghai")
    monkeypatch.setattr(timezone_utils, "get_localzone_name", detect)

    monkeypatch.setenv("TZ", "GMT")
    assert ApplicationConfig().timezone == "GMT"
    monkeypatch.setenv("TZ", "Europe/London")
    assert ApplicationConfig().timezone == "Europe/London"
    monkeypatch.delenv("TZ")
    assert ApplicationConfig().timezone == "Asia/Shanghai"
    detect.assert_called_once_with()


@pytest.mark.parametrize(
    "name",
    [
        "",
        "invalid/zone",
        ":Europe/London",
        "/usr/share/zoneinfo/America/New_York",
        "EST5EDT,M3.2.0,M11.1.0",
    ],
)
def test_application_timezone_rejects_unsupported_tz_without_caching(monkeypatch, name):
    """Invalid overrides fail explicitly and cannot affect the next application."""
    detect = Mock(return_value="Asia/Shanghai")
    monkeypatch.setattr(timezone_utils, "get_localzone_name", detect)
    monkeypatch.setenv("TZ", name)
    with pytest.raises(ValueError, match="TZ must be an IANA timezone name"):
        ApplicationConfig()
    with pytest.raises(ValueError, match="TZ must be an IANA timezone name"):
        ApplicationConfig(timezone=None)
    assert ApplicationConfig(timezone="Asia/Shanghai").timezone == "Asia/Shanghai"
    detect.assert_not_called()
    monkeypatch.delenv("TZ")
    assert ApplicationConfig().timezone == "Asia/Shanghai"
    detect.assert_called_once_with()


def test_explicit_application_timezone_takes_precedence(monkeypatch):
    """An explicit name or null bypasses automatic detection."""
    detect = Mock(return_value="America/New_York")
    monkeypatch.setattr(timezone_utils, "get_localzone_name", detect)

    assert ApplicationConfig(timezone="Europe/London").timezone == "Europe/London"
    context = ApplicationContext(timezone=None)
    assert context.app_config.timezone is None
    assert DailyWriteStep(app_context=context).config_value("daily_dir") == "daily"
    detect.assert_not_called()


def test_workspace_dir_expands_user_home(monkeypatch, tmp_path):
    """Home-relative workspace config is normalized before components consume it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    config = ApplicationConfig(workspace_dir="~/.copaw/workspaces/default")

    assert config.workspace_dir == str(tmp_path / ".copaw/workspaces/default")


def test_application_config_accepts_plugin_names_only():
    """Plugin configuration remains a simple list of installed entry-point names."""
    config = ApplicationConfig(plugins=["example"])

    assert config.plugins == ["example"]
    with pytest.raises(ValidationError):
        ApplicationConfig(plugins=[{"name": "example"}])


def test_application_config_accepts_plugin_defined_component_type():
    """Plugin component type names remain typed configuration buckets."""
    config = ApplicationConfig(
        components={
            "example.reranker": {
                "default": {"backend": "cross_encoder"},
            },
        },
    )

    assert config.components["example.reranker"]["default"].backend == "cross_encoder"


def test_application_config_rejects_unsafe_component_type():
    """Component type names cannot escape their workspace metadata directory."""
    with pytest.raises(ValidationError, match="Invalid component type"):
        ApplicationConfig(components={"../outside": {"default": {"backend": "unsafe"}}})


def test_dialog_dir_is_not_an_application_config_field():
    """The removed option is absent from schemas and ignored when supplied."""
    custom = ApplicationConfig(session_dir="sessions/", dialog_dir="somewhere/else")

    assert not hasattr(custom, "dialog_dir")
    assert "dialog_dir" not in custom.model_dump()
    assert "dialog_dir" not in ApplicationConfig.model_json_schema()["properties"]


@pytest.mark.parametrize("session_dir", ["/tmp/sessions", "C:\\sessions", "\\\\server\\share\\sessions"])
def test_application_rejects_absolute_session_dir_before_workspace_setup(tmp_path, session_dir):
    """Application initialization rejects session directories outside the workspace."""
    workspace = tmp_path / "workspace"

    with pytest.raises(ValidationError, match="session_dir must be a workspace-relative path"):
        Application(workspace_dir=str(workspace), session_dir=session_dir)

    assert not workspace.exists()


@pytest.mark.parametrize(
    "notebook_dir",
    ["", ".", "../outside", "notes/../outside", "~/notebook", "/tmp/notes", "C:\\notes"],
)
def test_notebook_dir_requires_a_workspace_subdirectory(notebook_dir):
    """Notebook configuration rejects paths that bypass the workspace subdirectory contract."""
    with pytest.raises(ValidationError, match="notebook_dir"):
        ApplicationConfig(notebook_dir=notebook_dir)


def test_dialog_watch_rule_follows_custom_session_dir(tmp_path):
    """Nested watch rules resolve the dialog directory below session_dir."""
    config = ApplicationConfig(session_dir="sessions")

    rules = build_watch_rules(
        config,
        tmp_path,
        watch_dirs=["session_dir/dialog"],
        watch_suffixes=["jsonl"],
    )

    assert len(rules) == 1
    assert rules[0].path == tmp_path / "sessions" / "dialog"
    assert rules[0].suffixes == ["jsonl"]


def test_dialog_watch_rule_matches_writer_for_normalized_session_dirs(tmp_path):
    """Nested watch rules use the same defaults and normalization as transcript writers."""
    for configured in ("", "parent//../sessions"):
        app_context = ApplicationContext(session_dir=configured)
        auto_memory = AutoMemoryStep(app_context=app_context)
        auto_memory.file_store = SimpleNamespace(workspace_path=tmp_path)

        rules = build_watch_rules(
            app_context.app_config,
            tmp_path,
            watch_dirs=["session_dir/dialog"],
            watch_suffixes=["jsonl"],
        )

        assert rules[0].path == auto_memory._session_path("s1").parent


def test_absolute_watch_rule_stays_outside_workspace(tmp_path):
    """Absolute literal watch directories retain their original location."""
    config = ApplicationConfig()
    absolute_dir = tmp_path.parent / "reme-dialogs"

    rules = build_watch_rules(
        config,
        tmp_path,
        watch_dirs=[str(absolute_dir)],
        watch_suffixes=["jsonl"],
    )

    assert rules[0].path == absolute_dir


def test_standard_session_paths_and_links_follow_custom_session_dir(tmp_path):
    """Writers derive transcript paths and links from session_dir."""
    app_context = ApplicationContext(session_dir="sessions")
    auto_memory = AutoMemoryStep(app_context=app_context)
    auto_memory.file_store = SimpleNamespace(workspace_path=tmp_path)
    daily_write = DailyWriteStep(app_context=app_context)

    assert auto_memory._session_path("s1") == tmp_path / "sessions" / "dialog" / "s1.jsonl"
    assert auto_memory._session_source_path("s1") == "sessions/dialog/s1.jsonl"
    assert auto_memory._session_link("s1") == "[[sessions/dialog/s1.jsonl]]"
    assert daily_write._session_link("s1") == "[[sessions/dialog/s1.jsonl]]"


def test_session_paths_and_links_normalize_custom_session_dir(tmp_path):
    """Writers and source links use the same normalized POSIX session root."""
    for configured, expected in (
        ("./sessions", "sessions"),
        (".", ""),
        ("parent//../sessions", "sessions"),
    ):
        app_context = ApplicationContext(session_dir=configured)
        auto_memory = AutoMemoryStep(app_context=app_context)
        auto_memory.file_store = SimpleNamespace(workspace_path=tmp_path)
        daily_write = DailyWriteStep(app_context=app_context)
        source_path = f"{expected + '/' if expected else ''}dialog/s1.jsonl"

        assert auto_memory._session_path("s1") == tmp_path / source_path
        assert auto_memory._session_source_path("s1") == source_path
        assert auto_memory._session_link("s1") == f"[[{source_path}]]"
        assert daily_write._session_link("s1") == f"[[{source_path}]]"


def test_dialog_classification_excludes_other_session_stores():
    """Only jsonl files below the dialog child directory are transcripts."""
    assert is_session_path("sessions/dialog/s1.jsonl", "sessions")
    assert not is_session_path("sessions/claude_code/s1.jsonl", "sessions")
    assert not is_session_path("session/dialog/s1.jsonl", "sessions")


def test_dialog_classification_normalizes_paths():
    """Session recognition compares normalized path representations."""
    assert is_session_path("sessions/dialog/s1.jsonl", "./sessions")
    assert is_session_path("./sessions//dialog/s1.jsonl", "sessions")
    assert is_session_path("dialog/s1.jsonl", ".")
    assert is_session_path("sessions/dialog/s1.jsonl", "parent/../sessions")


def test_session_rendering_follows_custom_session_dir():
    """Session-aware rendering recognizes a custom session root."""
    message = Msg(name="user", role="user", content=[{"type": "text", "text": "hello"}])
    chunk = FileChunk(
        path="sessions/dialog/s1.jsonl",
        start_line=1,
        end_line=1,
        text=message.model_dump_json(),
    )

    rendered = render_chunk_body(chunk, "sessions")

    assert rendered.startswith("[user @")
    assert rendered.endswith("] hello")
