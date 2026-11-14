"""HTTP extraction preserves complete sources and publishes only complete packages."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import asyncio
import json

import httpx
import pytest

from reme.components import ApplicationContext, RuntimeContext
from reme.schema import ResourceMetadata
from reme.steps.proactive.tools import web_extract
from reme.steps.proactive.tools.web_extract import WebExtractStep


@pytest.fixture
def mock_http(monkeypatch):
    requests = []
    client = httpx.AsyncClient

    def install(handler):
        def respond(request):
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(
            web_extract.httpx,
            "AsyncClient",
            lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs),
        )
        return requests

    return install


@pytest.mark.asyncio
async def test_html_preserves_full_sources_and_bounds_serialized_preview(tmp_path, mock_http):
    content = "完整网页内容。" * 2000
    html = (
        f"<html><title>原始来源</title><nav>Navigation</nav><article>{content}</article><script>script</script></html>"
    )
    mock_http(lambda _: httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"}))
    step = WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    answer = (await step(url="https://example.com/page", max_chars=10000)).answer
    manifest_path = tmp_path / answer["path"]
    manifest = ResourceMetadata.model_validate_json(manifest_path.read_text())
    assert manifest.source == "web_extract"
    assert {item.path for item in manifest.files} == {"source.html", "content.txt", "source.json"}
    assert (manifest_path.parent / "source.html").read_text() == html
    assert (manifest_path.parent / "content.txt").read_text() == content
    source = json.loads((manifest_path.parent / "source.json").read_text())
    assert source["title"] == "原始来源" and source["url"] == "https://example.com/page"
    assert source["fetched_at"] and source["download_bytes"] == len(html.encode())
    assert answer["truncated"] and content.startswith(answer["content"])
    assert len(json.dumps(answer, ensure_ascii=True).encode()) <= 8000
    assert all((tmp_path / path).is_file() for path in answer["paths"])
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data", "content_type", "expected"),
    [
        ('<meta charset="gb18030"><p>中文网页</p>'.encode("gb18030"), "text/html", "中文网页"),
        ('{"text":"中文"}'.encode("utf-16"), "application/json", '{"text":"中文"}'),
        (b"plain text", "text/plain; charset=unknown-charset", "plain text"),
    ],
)
async def test_declared_encodings_and_boms(tmp_path, mock_http, data, content_type, expected):
    mock_http(lambda _: httpx.Response(200, content=data, headers={"content-type": content_type}))
    answer = (
        await WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
            url="https://example.com/text",
        )
    ).answer
    assert answer["content"] == expected
    assert (tmp_path / answer["path"]).parent.joinpath("content.txt").read_text() == expected


@pytest.mark.asyncio
async def test_redirected_pdf_is_saved_without_claiming_text_extraction(tmp_path, mock_http):
    pdf = b"%PDF-1.7\nexample"

    def respond(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/source.pdf"})
        return httpx.Response(200, content=pdf, headers={"content-type": "application/pdf"})

    requests = mock_http(respond)
    answer = (
        await WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
            url="https://example.com/start",
        )
    ).answer
    directory = (tmp_path / answer["path"]).parent
    assert len(requests) == 2 and answer["final_url"] == "https://example.com/source.pdf"
    assert (directory / "source.pdf").read_bytes() == pdf
    assert not (directory / "content.txt").exists()
    assert answer["status"] == "downloaded" and answer["content"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [None, "", "file:///etc/passwd", "ftp://example.com", "https://user:pass@example.com"])
async def test_invalid_urls_fail_before_network(tmp_path, mock_http, url):
    requests = mock_http(lambda _: pytest.fail("invalid URL reached the network"))
    with pytest.raises(ValueError, match="url must"):
        await WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(url=url)
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("max_chars", [True, 0, 10001, "100"])
async def test_invalid_preview_limits_fail_before_network(tmp_path, mock_http, max_chars):
    requests = mock_http(lambda _: pytest.fail("invalid limit reached the network"))
    with pytest.raises(ValueError, match="max_chars"):
        await WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))(
            url="https://example.com",
            max_chars=max_chars,
        )
    assert not requests


@pytest.mark.asyncio
async def test_workspace_permission_is_checked_before_network(tmp_path, mock_http):
    requests = mock_http(lambda _: pytest.fail("forbidden path reached the network"))
    step = WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    with pytest.raises(PermissionError, match="resource package"):
        await step(RuntimeContext(_allowed_paths=["daily"]), url="https://example.com")
    assert not requests and not list(tmp_path.rglob("source.*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "data", "content_type", "max_bytes", "error", "message"),
    [
        (404, b"missing", "text/plain", 100, httpx.HTTPStatusError, "404"),
        (200, b"too much data", "text/plain", 5, ValueError, "exceeds"),
        (200, b"image", "image/png", 100, ValueError, "Unsupported content type"),
        (200, b"binary", "application/octet-stream", 100, ValueError, "not a PDF"),
    ],
)
async def test_failed_downloads_publish_no_partial_package(
    tmp_path,
    mock_http,
    status,
    data,
    content_type,
    max_bytes,
    error,
    message,
):
    mock_http(lambda _: httpx.Response(status, content=data, headers={"content-type": content_type}))
    step = WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)), max_file_bytes=max_bytes)
    with pytest.raises(error, match=message):
        await step(url="https://example.com")
    assert not list(tmp_path.rglob("metadata.json"))
    assert not list(tmp_path.glob(".reme-resource-*"))


@pytest.mark.asyncio
async def test_cancelled_download_publishes_nothing(tmp_path, mock_http):
    entered = asyncio.Event()

    async def respond(_):
        entered.set()
        await asyncio.Event().wait()

    mock_http(respond)
    step = WebExtractStep(app_context=ApplicationContext(workspace_dir=str(tmp_path)))
    task = asyncio.create_task(step(url="https://example.com"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list(tmp_path.rglob("metadata.json"))
