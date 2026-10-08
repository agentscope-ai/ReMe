"""Opt-in browser verification of the bundled view using an MCP host harness, not Codex screenshots.

Run with REME_CODEX_STATUS_UI=1 pytest tests/integration/test_codex_plugin_status_ui.py.
Requires Playwright and its Chromium browser; no model, service, or external network is used.
"""

# pylint: disable=missing-function-docstring,redefined-outer-name

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("REME_CODEX_STATUS_UI") != "1",
    reason="Opt-in status view browser check",
)
VIEW = Path(__file__).resolve().parents[2] / "integrations/codex/plugins/reme/ui/status.html"


def snapshot(language):
    return {
        "checked_at": 1790757300,
        "language": language,
        "plugin_version": "0.2.4",
        "mcpUrl": "http://localhost:2333/mcp",
        "service": {
            "health_check": {"reachable": True, "answer": "ReMe v0.4.1.13 - healthy"},
            "status": {
                "reachable": True,
                "answer": (
                    "Memory (estimated component object size)\n  file_store:default  1.25 MiB\n"
                    "  keyword_index:default  512 B\n  Components total  1.25 MiB\n  Process RSS       120.50 MiB"
                ),
            },
        },
        "auto_memory": {
            "enabled": True,
            "interval": 5,
            "queued_turns": 2,
            "queued_sessions": 1,
            "other_endpoint_queued_turns": 0,
        },
        "auto_recall": {"enabled": True},
        "auto_dream": {
            "enabled": True,
            "cron": "0 23 * * *",
            "timezone": "Asia/Shanghai",
            "phase": "running",
            "next_run_at": "2026-09-30T15:00:00+00:00",
            "last_run": None,
        },
        "recent_activity": [
            {"event": "memory_saved", "time": 1790757300, "turns": 1},
            {"event": "recall_found", "time": 1790757301, "turns": 0},
        ],
    }


def check_activity_history(page, frame, language, theme, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    dense = snapshot(language)
    dense["mcpUrl"] = "https://memory.example/" + "workspace/" * 40 + "mcp"
    dense["auto_dream"]["timezone"] = "America/Argentina/Buenos_Aires"
    dense["recent_activity"] = [
        {
            "event": "memory_failed" if index == 19 else "memory_saved",
            "time": 1790757300 + index,
            "turns": 1,
            **({"error": "ConnectionError_" + "detail" * 50} if index == 19 else {}),
        }
        for index in range(20)
    ]
    page.evaluate("value => window.result = {structuredContent:value}", dense)
    frame.locator("#refresh").click()
    frame.locator("#tab-memory").click()
    frame.locator("#activity-details > summary").click()
    playwright.expect(frame.locator("#activity li")).to_have_count(5)
    assert frame.locator("#activity-more").is_visible()
    assert frame.locator("#activity .activity-meta").first.is_hidden()
    frame.locator("#activity .activity-entry > summary").first.click()
    assert "ConnectionError_" in frame.locator("#activity .activity-meta").first.inner_text()
    assert frame.locator("#activity .activity-meta").first.is_visible()
    assert "GMT-3" in frame.locator("#activity time").first.get_attribute("title")
    for width in (320, 480, 768, 940):
        page.set_viewport_size({"width": width + 16, "height": 1250})
        page.evaluate("width => document.querySelector('iframe').style.width = `${width}px`", width)
        assert frame.locator("body").evaluate("node => node.scrollWidth <= window.innerWidth")
        assert frame.locator("#activity-details").evaluate("node => node.scrollWidth <= node.clientWidth")
        page.screenshot(path=str(tmp_path / f"status-test-host-{theme}-activity-{width}.png"), full_page=True)
        frame.locator("#tab-dream").click()
        assert frame.locator("body").evaluate("node => node.scrollWidth <= window.innerWidth")
        assert frame.locator("#panel-dream .card").evaluate_all(
            "nodes => nodes.every(node => node.scrollWidth <= node.clientWidth)",
        )
        frame.locator("#tab-memory").click()
    frame.locator("#activity-more").click()
    playwright.expect(frame.locator("#activity li")).to_have_count(20)
    assert frame.locator("#activity-more").get_attribute("aria-expanded") == "true"
    frame.locator("#refresh").click()
    playwright.expect(frame.locator("#activity li")).to_have_count(20)  # Preserve the chosen history length.
    assert frame.locator("#activity .activity-meta").first.is_visible()  # Preserve expanded event details.
    frame.locator("#activity-more").click()
    playwright.expect(frame.locator("#activity li")).to_have_count(5)
    assert frame.locator("#activity-more").get_attribute("aria-expanded") == "false"
    assert len(page.evaluate("window.calls.filter(m => m.method === 'tools/call')")) == 2
    frame.locator("#activity-details > summary").click()
    frame.locator("#tab-overview").click()


@pytest.mark.parametrize(("language", "theme"), [("en", "light"), ("zh", "dark")])
def test_initial_result_refresh_offline_failure_theme_and_untrusted_text(language, theme, tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    initial = snapshot(language)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch()
        page = browser.new_page(viewport={"width": 980, "height": 1250})
        page.set_default_timeout(5000)
        errors, external = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", lambda route: (external.append(route.request.url), route.abort()))
        page.set_content('<iframe title="MCP status test host" style="border:0;width:940px;height:1200px"></iframe>')
        page.evaluate(
            """({html, initial, theme}) => {
                window.calls = [];
                window.result = {structuredContent: initial};
                window.failRefresh = false;
                const iframe = document.querySelector('iframe');
                function send(value) { iframe.contentWindow.postMessage({jsonrpc: '2.0', ...value}, '*'); }
                window.sendStatus = () => send({method:'ui/notifications/tool-result', params:window.result});
                window.addEventListener('message', event => {
                    if (event.source !== iframe.contentWindow) return;
                    const m = event.data;
                    window.calls.push(m);
                    if (m.method === 'ui/initialize') {
                        send({id:m.id, result:{protocolVersion:'2026-01-26', hostCapabilities:{serverTools:{}},
                            hostContext:{theme, displayMode:'fullscreen', availableDisplayModes:['fullscreen']}}});
                    } else if (m.method === 'ui/notifications/initialized') {
                        window.sendStatus();
                    } else if (m.method === 'tools/call') {
                        if (window.failRefresh) send({id:m.id, error:{code:-32603, message:'test failure'}});
                        else send({id:m.id, result:window.result});
                    }
                });
                iframe.srcdoc = html;
            }""",
            {"html": VIEW.read_text(), "initial": initial, "theme": theme},
        )
        frame = page.frame_locator("iframe")
        try:
            frame.locator("#overview").wait_for(state="visible")
        except playwright.TimeoutError:
            assert not errors, errors
            raise
        assert frame.locator("html").get_attribute("data-theme") == theme
        assert "healthy" in frame.locator("#health").inner_text()
        assert "23:00" in frame.locator("#next").inner_text()
        assert frame.locator("#activity li").count() == 2
        assert not frame.locator("#connection-details").get_attribute("open")
        assert not frame.locator("#activity-details").get_attribute("open")
        assert not frame.locator("#view-help").get_attribute("open")
        assert not page.evaluate(
            "window.calls.some(m => m.method === 'tools/call')",
        )  # Opening reuses the supplied result.
        page.screenshot(path=str(tmp_path / f"status-test-host-{theme}.png"), full_page=True)
        for name in ("memory", "dream", "components"):
            frame.locator(f"#tab-{name}").click()
            assert frame.locator(f"#panel-{name}").is_visible()
            assert frame.locator("#panel-overview").is_hidden()
            page.screenshot(path=str(tmp_path / f"status-test-host-{theme}-{name}.png"), full_page=True)
        assert frame.locator("#rss").inner_text() == "120.50 MiB"
        assert frame.locator("#components .component").count() == 2
        assert frame.locator("#component-total").inner_text() == "1.25 MiB"
        frame.locator("#tab-components").press("Home")
        assert frame.locator("#panel-overview").is_visible()
        for width in (320, 480, 640, 768, 940):
            page.set_viewport_size({"width": width + 16, "height": 1250})
            page.evaluate("width => document.querySelector('iframe').style.width = `${width}px`", width)
            for name in ("overview", "memory", "dream", "components"):
                frame.locator(f"#tab-{name}").click()
                assert frame.locator("body").evaluate("node => node.scrollWidth <= window.innerWidth")
                assert frame.locator(f"#tab-{name}").evaluate(
                    "node => node.scrollWidth <= node.clientWidth && node.scrollHeight <= node.clientHeight",
                )
                assert frame.locator(f"#panel-{name} .card").evaluate_all(
                    "nodes => nodes.every(node => node.scrollWidth <= node.clientWidth)",
                )
                if width == 768 and name == "memory":
                    cards = frame.locator("#panel-memory article").all()
                    first, second = (card.bounding_box() for card in cards)
                    assert second["y"] >= first["y"] + first["height"]  # Avoid squeezed side-by-side cards.
            frame.locator("#tab-overview").click()
            if width == 480:
                page.screenshot(path=str(tmp_path / f"status-test-host-{theme}-narrow.png"), full_page=True)

        check_activity_history(page, frame, language, theme, tmp_path)

        offline = snapshot(language)
        offline["service"]["health_check"] = {"reachable": False, "error": "ConnectionError"}
        offline["service"]["status"] = {"reachable": False, "error": "ConnectionError"}
        offline["auto_memory"]["queued_turns"] = 3
        offline["mcpUrl"] = "<img src=x onerror=window.injected=true>"
        page.evaluate("value => window.result = {structuredContent:value}", offline)
        frame.locator("#refresh").click()
        playwright.expect(frame.locator("#connection-state")).to_have_attribute("data-tone", "bad")
        frame.locator("#connection-details summary").click()
        assert frame.locator("#endpoint").inner_text() == offline["mcpUrl"]
        assert frame.locator("#endpoint img").count() == 0
        assert "3" in frame.locator("#queue").inner_text()
        calls = page.evaluate("window.calls.filter(m => m.method === 'tools/call')")
        assert len(calls) == 3 and all(call["params"] == {"name": "reme_status", "arguments": {}} for call in calls)
        assert frame.locator("#error").is_hidden()  # Offline is a useful status result, not a broken panel.

        frame.locator("#tab-memory").click()
        frame.locator("#activity-details > summary").click()
        assert frame.locator("#activity li").first.is_visible()
        page.evaluate("window.failRefresh = true")
        frame.locator("#refresh").click()
        frame.locator("#error").wait_for(state="visible")
        assert "3" in frame.locator("#queue").inner_text()  # Failed refresh keeps the last snapshot.
        assert frame.locator("#refresh").is_enabled()
        assert frame.locator("#panel-memory").is_visible()  # Refresh preserves navigation.
        assert frame.locator("#activity li").first.is_visible()  # Refresh preserves expanded details.
        frame.locator("#tab-components").click()
        assert frame.locator("#rss").inner_text() == "—"
        assert frame.locator("#components .component").count() == 0
        assert not frame.locator("#service-details").get_attribute("open")
        frame.locator("#service-details summary").click()
        assert "ConnectionError" in frame.locator("#details").inner_text()
        unhealthy = snapshot(language)
        unhealthy["service"]["health_check"]["answer"] = "ReMe v0.4.1.13 - unhealthy"
        page.evaluate("value => { window.failRefresh = false; window.result = {structuredContent:value}; }", unhealthy)
        frame.locator("#tab-overview").click()
        frame.locator("#refresh").click()
        playwright.expect(frame.locator("#service-health")).to_have_attribute("data-tone", "bad")
        assert frame.locator("#connection-state").get_attribute("data-tone") == "good"  # Reachable is not healthy.
        assert not errors and not external
        browser.close()
