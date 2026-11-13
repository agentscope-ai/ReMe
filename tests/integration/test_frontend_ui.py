"""Opt-in Studio and documentation browser regressions with mocked service responses.

Run with REME_FRONTEND_UI=1 pytest tests/integration/test_frontend_ui.py.
REME_UI_BROWSER can select an installed Chromium-compatible browser. No model or external service is used.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("REME_FRONTEND_UI") != "1", reason="Opt-in frontend browser checks")
ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "notes/data.csv": 'name,description,note\r\nAlice,"first\nsecond","a ""quote"""\r\nBob,end,  ',
    "notes/metadata.md": (
        "---\ntitle: Research\ndescription: |\n  first line\n  second line\n"
        "memory_tags:\n  - agents\n  - memory\n---\n# Body"
    ),
}


@pytest.fixture(scope="module")
def browser():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as runtime:
        instance = runtime.chromium.launch(executable_path=os.environ.get("REME_UI_BROWSER"))
        yield instance
        instance.close()


def serve(project, command, tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    output = tmp_path / "server.log"
    with (
        output.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            [*command, "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
            cwd=ROOT / project,
            env={**os.environ, "VITE_REME_API_URL": "/", "VITE_REME_WORKSPACE_EXTENSIONS": "md,txt,csv"},
            stdout=log,
            stderr=subprocess.STDOUT,
        ) as process,
    ):
        url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(output.read_text(encoding="utf-8"))
                try:
                    with urllib.request.urlopen(url, timeout=1):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                pytest.fail(output.read_text(encoding="utf-8"))
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(scope="module")
def studio_url(tmp_path_factory):
    yield from serve(
        "reme_studio",
        ["node_modules/.bin/vite", "--config", "vite.static.config.ts"],
        tmp_path_factory.mktemp("studio-ui"),
    )


@pytest.fixture(scope="module")
def docs_url(tmp_path_factory):
    yield from serve(
        "github-pages",
        ["node_modules/.bin/vitepress", "dev", ".generated/site"],
        tmp_path_factory.mktemp("docs-ui"),
    )


@pytest.fixture
def studio_page(browser, studio_url):
    context = browser.new_context(locale="zh-CN", viewport={"width": 1280, "height": 900})
    page = context.new_page()
    errors, queries = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))

    def service(route):
        request = route.request
        if request.method != "POST":
            route.continue_()
            return
        action = request.url.rsplit("/", 1)[-1]
        body = request.post_data_json
        metadata = {}
        if action == "app_config":
            answer = {
                "workspace_dir": "/mock/workspace",
                "daily_dir": "daily",
                "digest_dir": "digest",
            }
        elif action == "list":
            answer, metadata = "files", {"items": list(FILES)}
        elif action == "load":
            answer, metadata = FILES[body["path"]], {"mtime": "2026-11-10T00:00:00Z"}
        elif action == "session_read":
            route.fulfill(json={"success": False, "metadata": {"code": "session_not_supported"}})
            return
        elif action == "chat":
            queries.append(body["query"])
            events = [
                {
                    "chunk_type": "reply_start",
                    "chunk": None,
                    "status": "running",
                    "session_id": "ui-session",
                    "turn_id": "ui-turn",
                    "done": False,
                },
                {"chunk_type": "content", "chunk": "已收到", "done": False},
                {
                    "chunk_type": "reply_end",
                    "chunk": {"status": "completed", "result": "已收到"},
                    "status": "completed",
                    "event": "end",
                    "done": False,
                },
            ]
            content = (
                "".join(f"data:{json.dumps(event, ensure_ascii=False)}\n\n" for event in events) + "data:[DONE]\n\n"
            )
            route.fulfill(content_type="text/event-stream", body=content)
            return
        else:
            answer = "ok"
        route.fulfill(json={"success": True, "answer": answer, "metadata": metadata})

    page.route("**/*", service)
    yield page, studio_url, errors, queries
    context.close()


def test_studio_ime_confirmation_and_repeated_send(studio_page):
    page, url, errors, queries = studio_page
    page.goto(url)
    page.locator(".welcome button").click()
    composer = page.locator(".chat textarea")
    composer.fill("正在输入中文")
    composer.evaluate(
        "node => node.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', isComposing:true, bubbles:true}))",
    )
    assert queries == []
    assert composer.input_value() == "正在输入中文"
    composer.evaluate("""node => {
        node.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
        node.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', bubbles:true}));
    }""")
    page.get_by_text("已收到", exact=True).wait_for()
    assert queries == ["正在输入中文"]
    assert errors == []


def test_studio_storage_denial_preserves_usable_ui(studio_page):
    page, url, errors, _ = studio_page
    page.add_init_script("""for (const method of ['getItem','setItem','removeItem']) {
        Storage.prototype[method] = () => { throw new Error('Browser storage blocked'); };
    }""")
    page.goto(url)
    page.locator(".theme-menu summary").click()
    page.locator('.theme-menu [role="menuitemradio"]').nth(1).click()
    assert page.locator("html").get_attribute("data-theme") == "dark"
    page.locator(".welcome button").click()
    page.locator(".chat textarea").fill("草稿保留在内存")
    assert page.locator(".chat textarea").input_value() == "草稿保留在内存"
    assert errors == []


def test_studio_structured_previews_and_cancelled_resize(studio_page):
    page, url, errors, _ = studio_page
    page.goto(url)
    page.locator('button[title="notes/data.csv"]').click()
    table = page.locator("table")
    table.locator("tbody tr").nth(1).wait_for()
    assert table.locator("tbody tr").count() == 2
    assert table.locator("tbody tr").first.locator("td").nth(1).text_content() == "first\nsecond"
    page.locator('button[title="notes/metadata.md"]').click()
    preview = page.locator(".content dl")
    assert "agents" in preview.inner_text()
    assert "second line" in preview.inner_text()
    assert page.locator('.tabs [role="tab"][aria-selected="true"]').count() == 1
    separator = page.locator('[role="separator"]')
    separator.evaluate("""node => {
        node.dispatchEvent(new PointerEvent('pointerdown', {button:0, clientX:260, bubbles:true}));
        window.dispatchEvent(new PointerEvent('pointermove', {clientX:380}));
        window.dispatchEvent(new PointerEvent('pointercancel', {clientX:0}));
    }""")
    assert page.locator("main.shell").evaluate("node => node.style.getPropertyValue('--navigator-width')") == "380px"
    assert "navigator-is-resizing" not in page.locator("main.shell").get_attribute("class")
    assert errors == []


def test_docs_clipboard_failure_and_route_change(browser, docs_url):
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.route("https://**/*", lambda route: route.abort())
    page.goto(docs_url + "/en/overview")
    button = page.locator(".copy-markdown")
    button.wait_for()
    page.evaluate("""() => Object.defineProperty(navigator, 'clipboard', {configurable:true, value:{
        writeText: async () => { throw new Error('Permission denied'); }
    }})""")
    button.click()
    page.get_by_text("Copy failed. Try again", exact=True).wait_for()
    page.evaluate(
        """() => { navigator.clipboard.writeText = () => new Promise(resolve => { window.finishCopy = resolve; }); }""",
    )
    button.click()
    page.locator('a[href*="/en/memory_as_file"]').first.click()
    page.wait_for_url("**/en/memory_as_file*")
    page.locator("h1#memory-as-file").wait_for()
    page.evaluate("() => window.finishCopy()")
    assert page.locator(".copy-markdown").inner_text().strip() == "Copy Markdown"
    assert not errors
    context.close()
