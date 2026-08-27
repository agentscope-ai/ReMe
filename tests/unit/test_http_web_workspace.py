"""HTTP service coverage for MCP and the optional bundled web workspace."""

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from reme.components.service.http_service import HttpService
from reme.components.job import BaseJob, StreamJob
from reme.utils import REME_WEB_STATIC_DIR, resolve_web_static_dir


class _FakeApplication:
    def __init__(self) -> None:
        self.config = SimpleNamespace(app_name="ReMe test")
        self.started = False
        self.closed = False

    async def start(self) -> None:
        """Record that the application lifespan started."""
        self.started = True

    async def close(self) -> None:
        """Record that the application lifespan closed."""
        self.closed = True


def _static_build(tmp_path: Path) -> Path:
    static_dir = tmp_path / "web"
    assets_dir = static_dir / "assets"
    assets_dir.mkdir(parents=True)
    (static_dir / "index.html").write_text(
        "<main>ReMe workspace</main>",
        encoding="utf-8",
    )
    (static_dir / "favicon.svg").write_text("<svg></svg>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('reme')", encoding="utf-8")
    return static_dir


def test_http_service_serves_workspace_without_shadowing_jobs(tmp_path: Path) -> None:
    """Serve workspace files and preserve explicitly registered job routes."""
    static_dir = _static_build(tmp_path)
    app = _FakeApplication()
    service = HttpService(web_static_dir=str(static_dir))
    service.build_service(app)  # type: ignore[arg-type]

    @service.service.post("/status")
    async def status():
        return {"success": True}

    service.finalize_service(app)  # type: ignore[arg-type]

    with TestClient(service.service) as client:
        assert client.get("/").text == "<main>ReMe workspace</main>"
        assert client.get("/memory/topic").text == "<main>ReMe workspace</main>"
        assert client.get("/favicon.svg").text == "<svg></svg>"
        assert client.get("/assets/app.js").text == "console.log('reme')"
        assert client.post("/status").json() == {"success": True}
        assert client.get("/status").status_code == 405
        assert client.get("/status").headers["allow"] == "POST"
        assert client.get("/").headers["cache-control"] == "no-cache, no-store, must-revalidate"

    assert app.started is True
    assert app.closed is True


def test_http_service_can_disable_workspace(tmp_path: Path) -> None:
    """Leave the root route unregistered when workspace serving is disabled."""
    app = _FakeApplication()
    service = HttpService(
        web_enabled=False,
        web_static_dir=str(_static_build(tmp_path)),
    )
    service.build_service(app)  # type: ignore[arg-type]
    service.finalize_service(app)  # type: ignore[arg-type]

    with TestClient(service.service) as client:
        assert client.get("/").status_code == 404


def test_http_service_exposes_non_stream_jobs_as_mcp_tools() -> None:
    """The HTTP backend exposes the same non-stream Job instance through MCP."""

    async def run() -> None:
        service = HttpService(web_enabled=False)
        service.build_service(_FakeApplication())  # type: ignore[arg-type]
        job = BaseJob(name="search", description="Search memories")

        assert service.add_job(job) is True
        assert service.mcp_server is not None
        assert await service.mcp_server.get_tool("search") is not None
        assert any(route.path == "/search" for route in service.service.routes)

    asyncio.run(run())


def test_http_service_skips_stream_jobs_for_mcp() -> None:
    """Stream jobs remain available over HTTP SSE without becoming MCP tools."""

    async def run() -> None:
        service = HttpService(web_enabled=False)
        service.build_service(_FakeApplication())  # type: ignore[arg-type]

        assert service.add_job(StreamJob(name="stream")) is True
        assert service.mcp_server is not None
        assert await service.mcp_server.get_tool("stream") is None
        assert any(route.path == "/stream" for route in service.service.routes)

    asyncio.run(run())


def test_http_service_uses_exact_mcp_path_and_runs_one_application_lifespan() -> None:
    """Serve MCP at /mcp while starting and closing the shared Application once."""

    class CountingApplication(_FakeApplication):
        """Record how often the shared Application lifecycle is entered."""

        def __init__(self) -> None:
            super().__init__()
            self.start_count = 0
            self.close_count = 0

        async def start(self) -> None:
            self.start_count += 1

        async def close(self) -> None:
            self.close_count += 1

    app = CountingApplication()
    service = HttpService(web_enabled=False)
    service.build_service(app)  # type: ignore[arg-type]
    mcp_server = service.mcp_server

    class VerifyFastMCPAppMiddleware(BaseHTTPMiddleware):
        """Prove requests retain FastMCP middleware and application state."""

        async def dispatch(self, request, call_next):
            """Verify the child app context and mark its response."""
            assert request.app.state.fastmcp_server is mcp_server
            response = await call_next(request)
            response.headers["X-FastMCP-Middleware"] = "preserved"
            return response

    service.mcp_app.add_middleware(VerifyFastMCPAppMiddleware)

    with TestClient(service.service, follow_redirects=False) as client:
        response = client.post(
            "/mcp",
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )
        assert response.status_code == 200
        assert '"serverInfo"' in response.text
        assert response.headers["X-FastMCP-Middleware"] == "preserved"
        assert client.post("/mcp/mcp", json={}).status_code == 404

    assert app.start_count == 1
    assert app.close_count == 1


def test_http_service_can_disable_mcp() -> None:
    """Allow deployments to retain the legacy HTTP-only surface explicitly."""
    service = HttpService(web_enabled=False, mcp_enabled=False)
    service.build_service(_FakeApplication())  # type: ignore[arg-type]

    with TestClient(service.service) as client:
        assert client.post("/mcp", json={}).status_code == 404


def test_http_service_rejects_invalid_or_conflicting_mcp_paths() -> None:
    """Reject paths that shadow built-ins and jobs that shadow the MCP endpoint."""
    for path in (
        "mcp",
        "/",
        "/mcp/",
        "/docs",
        "/{rest:path}",
        "/mcp?mode=test",
        "/mcp#fragment",
        "/mcp path",
        "/mcp//nested",
        "/mcp/../nested",
    ):
        with pytest.raises(ValueError, match="mcp_path"):
            HttpService(mcp_path=path)

    service = HttpService(web_enabled=False)
    service.build_service(_FakeApplication())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conflicts with the MCP endpoint"):
        service.add_job(BaseJob(name="mcp"))


def test_http_service_fails_startup_preflight_for_mcp_job_conflict() -> None:
    """Do not let BaseService's tolerant registration hide reserved-route conflicts."""
    service = HttpService(web_enabled=False)
    service.build_service(_FakeApplication())  # type: ignore[arg-type]
    app = SimpleNamespace(
        context=SimpleNamespace(jobs={"mcp": BaseJob(name="mcp")}),
    )

    with pytest.raises(ValueError, match="conflicts with the MCP endpoint"):
        service.add_jobs(app)


def test_http_service_does_not_serve_symlinks_outside_static_dir(
    tmp_path: Path,
) -> None:
    """Do not expose files reached through symlinks outside the static build."""
    static_dir = _static_build(tmp_path)
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("private", encoding="utf-8")
    try:
        (static_dir / "escape.txt").symlink_to(secret_file)
    except OSError as error:
        pytest.skip(f"Symbolic links are unavailable: {error}")

    app = _FakeApplication()
    service = HttpService(web_static_dir=str(static_dir))
    service.build_service(app)  # type: ignore[arg-type]
    service.finalize_service(app)  # type: ignore[arg-type]

    with TestClient(service.service) as client:
        assert client.get("/escape.txt").text == "<main>ReMe workspace</main>"


def test_static_dir_configuration_precedes_environment(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Prefer an explicit static directory over the environment setting."""
    configured = _static_build(tmp_path / "configured")
    environment = _static_build(tmp_path / "environment")
    monkeypatch.setenv(REME_WEB_STATIC_DIR, str(environment))

    assert resolve_web_static_dir(str(configured)) == configured.resolve()
    assert resolve_web_static_dir() == environment.resolve()


def test_static_dir_uses_optional_studio_package(monkeypatch, tmp_path: Path) -> None:
    """Use static assets supplied by the separately installed Studio wheel."""
    static_dir = _static_build(tmp_path / "studio")
    studio = ModuleType("reme_studio")
    studio.static_dir = lambda: static_dir  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "reme_studio", studio)

    assert resolve_web_static_dir() == static_dir.resolve()
