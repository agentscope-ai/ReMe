"""HTTP service coverage for the optional bundled web workspace."""

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from reme.components.service.http_service import HttpService
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
    (static_dir / "index.html").write_text("<main>ReMe workspace</main>", encoding="utf-8")
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
        assert client.get("/").headers["cache-control"] == "no-cache, no-store, must-revalidate"

    assert app.started is True
    assert app.closed is True


def test_http_service_can_disable_workspace(tmp_path: Path) -> None:
    """Leave the root route unregistered when workspace serving is disabled."""
    app = _FakeApplication()
    service = HttpService(web_enabled=False, web_static_dir=str(_static_build(tmp_path)))
    service.build_service(app)  # type: ignore[arg-type]
    service.finalize_service(app)  # type: ignore[arg-type]

    with TestClient(service.service) as client:
        assert client.get("/").status_code == 404


def test_static_dir_configuration_precedes_environment(monkeypatch, tmp_path: Path) -> None:
    """Prefer an explicit static directory over the environment setting."""
    configured = _static_build(tmp_path / "configured")
    environment = _static_build(tmp_path / "environment")
    monkeypatch.setenv(REME_WEB_STATIC_DIR, str(environment))

    assert resolve_web_static_dir(str(configured)) == configured.resolve()
    assert resolve_web_static_dir() == environment.resolve()
