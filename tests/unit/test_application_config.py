"""Application path configuration contracts."""

# pylint: disable=protected-access

from types import SimpleNamespace

from agentscope.message import Msg

from reme.components import ApplicationContext
from reme.schema import ApplicationConfig, FileChunk
from reme.steps.evolve.auto_memory import AutoMemoryStep
from reme.steps.file_io.daily_write import DailyWriteStep
from reme.steps.index._source_format import is_session_path, render_chunk_body
from reme.steps.index._watch_rules import build_watch_rules


def test_dialog_dir_is_not_an_application_config_field():
    """The removed option is absent from schemas and ignored when supplied."""
    custom = ApplicationConfig(session_dir="sessions/", dialog_dir="somewhere/else")

    assert not hasattr(custom, "dialog_dir")
    assert "dialog_dir" not in custom.model_dump()
    assert "dialog_dir" not in ApplicationConfig.model_json_schema()["properties"]


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
