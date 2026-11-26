"""Stable public address for Studio, management and the replaceable HTTP service."""

import asyncio
import hashlib
import ipaddress
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..components.service.web_static import resolve_web_static_dir
from ..constants import REME_DEFAULT_HOST, REME_DEFAULT_PORT
from .config import SettingsRequest
from .storage import write_private
from .supervisor import OperationConflict, Supervisor, lock_directory

_HOP_HEADERS = frozenset(
    (
        b"connection",
        b"keep-alive",
        b"proxy-authenticate",
        b"proxy-authorization",
        b"te",
        b"trailer",
        b"transfer-encoding",
        b"upgrade",
    ),
)


def _forward_headers(headers):
    excluded = _HOP_HEADERS | {
        name.strip().lower() for key, value in headers if key.lower() == b"connection" for name in value.split(b",")
    }
    return [(key, value) for key, value in headers if key.lower() not in excluded]


def _loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_gateway(
    supervisor: Supervisor,
    static_dir: Path | None,
    *,
    local_listener: bool = True,
    admin_token: str = "",
    published_loopback: bool = False,
    remote_management: bool = False,
) -> FastAPI:
    """Bind management and static resources independently of the child process."""
    transport = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5), trust_env=False)

    @asynccontextmanager
    async def lifespan(_app):
        watch = asyncio.create_task(supervisor.watch())
        try:
            yield
        finally:
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
            await supervisor.close()
            await transport.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    async def check_management_request(request: Request):
        peer = request.client.host if request.client else None
        local_host = _loopback(request.url.hostname)
        local = local_host and local_listener and _loopback(peer)
        if not local and not (published_loopback and local_host):
            if request.url.scheme != "https":
                raise HTTPException(403, "Remote ReMe management requires HTTPS")
            if not remote_management or not admin_token:
                raise HTTPException(403, "Remote ReMe management is disabled")
        if admin_token or not local:
            supplied = request.headers.get("x-reme-admin-token", "").encode("utf-8")
            if not admin_token or not secrets.compare_digest(supplied, admin_token.encode("utf-8")):
                raise HTTPException(401, "A ReMe administrator token is required")
        origin = request.headers.get("origin")
        if origin is None:
            if request.headers.get("sec-fetch-site") == "cross-site":
                raise HTTPException(403, "Cross-origin ReMe management is unavailable")
        elif origin not in {str(request.base_url).rstrip("/"), "https://appassets.androidplatform.net"}:
            raise HTTPException(403, "Cross-origin ReMe management is unavailable")

    management = [Depends(check_management_request)]

    @app.exception_handler(RequestValidationError)
    async def invalid_settings(_request, error):
        fields = sorted({str(item["loc"][-1]) for item in error.errors() if item["loc"]})
        return JSONResponse({"detail": f"Invalid settings: {', '.join(fields)}"}, status_code=422)

    @app.exception_handler(OperationConflict)
    async def conflict(_request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.get("/admin/config", dependencies=management)
    async def settings():
        return JSONResponse(supervisor.settings(), headers={"Cache-Control": "no-store"})

    @app.get("/admin/status", dependencies=management)
    async def status(operation_id: str | None = None):
        return JSONResponse(supervisor.status(operation_id), headers={"Cache-Control": "no-store"})

    async def submit(settings: SettingsRequest | None):
        try:
            return await supervisor.request(settings)
        except (RuntimeError, OSError, ValueError, TimeoutError) as error:
            if isinstance(error, OperationConflict):
                raise
            raise HTTPException(400, str(error) or "ReMe configuration check timed out") from None

    @app.post("/admin/apply", dependencies=management, status_code=202)
    async def apply(settings: SettingsRequest):
        return await submit(settings)

    @app.post("/admin/restart", dependencies=management, status_code=202)
    async def restart():
        return await submit(None)

    @app.post("/admin/test_model", dependencies=management)
    async def test_model(settings: SettingsRequest):
        try:
            await supervisor.test_model(settings)
        except (RuntimeError, OSError, ValueError, TimeoutError) as error:
            if isinstance(error, OperationConflict):
                raise
            raise HTTPException(400, str(error) or "Model connection test timed out") from None
        return {"success": True}

    @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def forward(request: Request, path: str):
        if path == "admin" or path.startswith("admin/"):
            raise HTTPException(404)
        if static_dir is not None and request.method in {"GET", "HEAD"}:
            target = (static_dir / (path or "index.html")).resolve()
            if target.is_relative_to(static_dir) and target.is_file():
                return FileResponse(target, headers={"Cache-Control": "no-cache"} if not path else None)
        port = supervisor.port
        unavailable = HTTPException(503, "ReMe is starting or unavailable", headers={"Retry-After": "1"})
        if port is None:
            raise unavailable
        raw_path = request.scope["raw_path"]
        query = request.scope["query_string"]
        target = httpx.URL(f"http://127.0.0.1:{port}").copy_with(raw_path=raw_path + (b"?" + query if query else b""))
        headers = [
            (key, value) for key, value in _forward_headers(request.headers.raw) if not key.startswith(b"x-forwarded-")
        ]
        if request.client:
            headers.append((b"x-forwarded-for", request.client.host.encode()))
        headers.append((b"x-forwarded-proto", request.url.scheme.encode()))
        downstream = httpx.Request(request.method, target, headers=headers, content=request.stream())
        try:
            upstream = await transport.send(downstream, stream=True)
        except httpx.RequestError:
            raise unavailable from None

        async def body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        response = StreamingResponse(body(), status_code=upstream.status_code)
        response.raw_headers = _forward_headers(upstream.headers.raw)
        return response

    return app


def run_supervised(arguments: dict, service: dict, inherited: dict[str, str]) -> None:
    """Run the HTTP supervisor in the foreground so a host manager can own it."""
    host = service.get("host") or REME_DEFAULT_HOST
    port = int(service.get("port") or REME_DEFAULT_PORT)
    instance = (str(Path.cwd()), arguments.get("config") or "default", host, port)
    identity = hashlib.sha256(json.dumps(instance).encode()).hexdigest()[:16]
    fallback = Path.home() / ".config" / "reme" / "instances" / identity
    directory = Path(os.getenv("REME_DAEMON_DIR") or fallback).expanduser().resolve()
    static_dir = resolve_web_static_dir(service.get("web_static_dir")) if service.get("web_enabled") else None
    with lock_directory(directory):
        explicit_token = os.getenv("REME_ADMIN_TOKEN", "").strip()
        published_loopback = _loopback(os.getenv("REME_ADMIN_PUBLISHED_HOST", "").strip("[]"))
        token = explicit_token
        if published_loopback and not token:
            token_path = directory / "admin-token"
            if not token_path.is_file():
                write_private(token_path, secrets.token_urlsafe(32) + "\n")
            token = token_path.read_text(encoding="utf-8").strip()
        supervisor = Supervisor(directory, arguments, inherited)
        transport_options = {
            key: value
            for key, value in service.items()
            if key.startswith("ssl_") or key in {"forwarded_allow_ips", "proxy_headers"}
        }
        app = build_gateway(
            supervisor,
            static_dir,
            local_listener=_loopback(host),
            admin_token=token,
            published_loopback=published_loopback,
            remote_management=bool(explicit_token),
        )
        uvicorn.run(app, host=host, port=port, timeout_graceful_shutdown=5, **transport_options)
