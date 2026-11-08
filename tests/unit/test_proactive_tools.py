"""Connector errors and JSON parsing through native Steps and Jobs."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import json

import pytest

from reme.components.application_context import ApplicationContext
from reme.components.job import BaseJob
from reme.config import resolve_app_config
from reme.steps.proactive.data_connectors import FinanceNewsStep
from reme.steps.proactive.tools import CheckJsonStep


@pytest.fixture
def app(tmp_path):
    return ApplicationContext(workspace_dir=str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [{"hello": "主人"}, [1, "two"], "text", 42, 1.5, True, False, None])
async def test_check_json_returns_every_json_value_without_modifying_file(app, tmp_path, value):
    path = tmp_path / "value.json"
    original = json.dumps(value, ensure_ascii=False).encode()
    path.write_bytes(original)
    result = await CheckJsonStep(app_context=app)(path="value.json")
    assert result.success and result.answer == value
    assert path.read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ['{\n  "value": 1,\n}\n', '{\n  "value": nope\n}\n'])
async def test_parse_error_contains_path_line_column_and_error(app, tmp_path, content):
    (tmp_path / "broken.json").write_text(content)
    result = await CheckJsonStep(app_context=app)(path="broken.json")
    assert not result.success
    assert result.answer["path"] == "broken.json"
    with pytest.raises(json.JSONDecodeError) as expected:
        json.loads(content)
    assert result.answer["line"] == expected.value.lineno
    assert result.answer["column"] == expected.value.colno
    assert result.answer["error"] == str(expected.value)
    assert (tmp_path / "broken.json").read_text() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-32"])
async def test_json_encoding_detection_supports_boms(app, tmp_path, encoding):
    (tmp_path / "encoded.json").write_bytes('{"name":"主人"}'.encode(encoding))
    result = await CheckJsonStep(app_context=app)(path="encoded.json")
    assert result.success and result.answer == {"name": "主人"}


@pytest.mark.asyncio
async def test_invalid_encoding_fails_without_replacement_characters(app, tmp_path):
    (tmp_path / "invalid.json").write_bytes(b'{"value":"\xff"}')
    result = await CheckJsonStep(app_context=app)(path="invalid.json")
    assert not result.success and result.answer["line"] is None
    assert "decode" in result.answer["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["missing.json", "../outside.json", "~/value.json", ""])
async def test_path_and_io_errors_are_structured(app, path):
    result = await CheckJsonStep(app_context=app)(path=path)
    assert not result.success
    assert result.answer["path"] == path
    assert result.answer["line"] is None and result.answer["column"] is None
    assert result.answer["error"]


@pytest.mark.asyncio
async def test_json_does_not_parse_a_truncated_valid_prefix(app, tmp_path):
    content = b"{}" + b" " * 20 + b"invalid tail"
    (tmp_path / "large.json").write_bytes(content)
    result = await CheckJsonStep(app_context=app, max_file_bytes=2)(path="large.json")
    assert not result.success and "read limit" in result.answer["error"]
    assert (tmp_path / "large.json").read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [[], ["daily"], ["../outside"]])
async def test_json_respects_allowed_paths(app, tmp_path, scope):
    (tmp_path / "value.json").write_text("{}")
    result = await CheckJsonStep(app_context=app)(path="value.json", _allowed_paths=scope)
    assert not result.success and "permission" in result.answer["error"]


@pytest.mark.asyncio
async def test_json_rejects_a_symlink_escaping_the_workspace(app, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}")
    try:
        (tmp_path / "value.json").symlink_to(outside)
        result = await CheckJsonStep(app_context=app)(path="value.json")
        assert not result.success and "workspace" in result.answer["error"]
    finally:
        outside.unlink()


@pytest.mark.asyncio
async def test_finance_source_failure_publishes_no_resources(app, tmp_path, monkeypatch):
    def unavailable(_step, _client, _cursor):
        raise ValueError("CLS source unavailable")

    monkeypatch.setattr(FinanceNewsStep, "_request_page", unavailable)
    with pytest.raises(RuntimeError, match="CLS source unavailable"):
        await FinanceNewsStep(app_context=app, request_interval=0, max_retries=1)()
    assert not list(tmp_path.glob(".reme-resource-*"))
    assert not (tmp_path / "resource").exists()


@pytest.mark.asyncio
async def test_new_tools_run_through_registered_native_jobs(app, tmp_path, monkeypatch):
    monkeypatch.setattr(FinanceNewsStep, "_fetch_recent", lambda _step, _end: [])
    jobs = resolve_app_config(config="proactive", log_config=False)["jobs"]
    assert jobs["pa_chat"]["steps"][0]["job_tools"] == [
        "screen_capture",
        "check_json",
        "write_proactive_inbox",
        "manage_temp_proactive_agent",
    ]
    for name in ["finance_news", "check_json"]:
        job = BaseJob(name=name, app_context=app, **jobs[name])
        async with job:
            if name == "check_json":
                (tmp_path / "value.json").write_text("null")
                response = await job(path="value.json")
                assert response.success and response.answer is None
            else:
                response = await job()
                assert response.success
                assert set(response.answer) == {"context", "paths"}
                assert response.answer["paths"] == []
