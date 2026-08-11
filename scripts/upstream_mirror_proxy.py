#!/usr/bin/env python3
"""Expose fixed upstream HTTP(S) services through path-prefixed mirror URLs.

The proxy is intentionally not a general-purpose forward proxy: callers can
only reach upstreams configured by the operator.  It uses only the Python
standard library so it can run on a small relay host without installing extra
packages.

Examples:
    python3 scripts/upstream_mirror_proxy.py
    python3 scripts/upstream_mirror_proxy.py --bind 0.0.0.0 --allow 192.0.2.0/24
    python3 scripts/upstream_mirror_proxy.py --route pypi=https://pypi.org

With the default routes, configure clients with base URLs such as:
    HF_MIRROR_URL=http://relay-host:18080/hf
    ARXIV_MIRROR_URL=http://relay-host:18080/arxiv
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import signal
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

DEFAULT_ROUTES = {
    "hf": "https://huggingface.co",
    "arxiv": "https://arxiv.org",
}
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_TIMEOUT = 600.0
DEFAULT_MAX_CONCURRENCY = 16
BUFFER_SIZE = 64 * 1024

REQUEST_HEADERS = {
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "if-match",
    "if-modified-since",
    "if-none-match",
    "if-range",
    "if-unmodified-since",
    "range",
    "user-agent",
}
RESPONSE_HEADERS_TO_SKIP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class ProxyConfig:
    """Immutable configuration shared by request handler threads."""

    routes: dict[str, str]
    allowed_clients: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]
    timeout: float
    slots: threading.BoundedSemaphore


def parse_route(value: str) -> tuple[str, str]:
    """Parse and validate one NAME=URL route."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("route must have the form NAME=URL")
    name, upstream = value.split("=", maxsplit=1)
    name, upstream = name.strip().strip("/"), upstream.strip().rstrip("/")
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
        raise argparse.ArgumentTypeError("route name may contain only letters, numbers, '_' and '-'")

    parsed = urlsplit(upstream)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("route upstream must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("route upstream must not contain credentials, a query, or a fragment")
    return name, upstream


def parse_network(value: str) -> ipaddress.IPv4Network | ipaddress.IPv6Network:
    """Parse one allowed address or CIDR."""
    try:
        return ipaddress.ip_network(value, strict=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def client_is_allowed(
    address: str,
    networks: Iterable[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Return whether an address belongs to any configured network."""
    try:
        client = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(client in network for network in networks)


class MirrorProxyHandler(BaseHTTPRequestHandler):
    """Proxy GET and HEAD requests to a fixed, path-selected upstream."""

    protocol_version = "HTTP/1.0"
    server_version = "ReMeMirrorProxy/1.0"

    @property
    def config(self) -> ProxyConfig:
        """Return the server-wide immutable proxy configuration."""
        return self.server.proxy_config  # type: ignore[attr-defined,no-any-return]

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy a GET request."""
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Proxy a HEAD request."""
        self._handle_request(send_body=False)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        """Reject unsupported methods."""
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Only GET and HEAD are supported")

    do_PUT = do_POST
    do_PATCH = do_POST
    do_DELETE = do_POST
    do_CONNECT = do_POST

    def _handle_request(self, *, send_body: bool) -> None:
        started = time.monotonic()
        if not client_is_allowed(self.client_address[0], self.config.allowed_clients):
            logging.warning("rejected client address=%s path=%s", self.client_address[0], self.path)
            self.send_error(HTTPStatus.FORBIDDEN, "Client address is not allowed")
            return

        parsed = urlsplit(self.path)
        if parsed.path == "/healthz":
            self._send_text(HTTPStatus.OK, "ok\n", send_body=send_body)
            return
        if parsed.path == "/routes":
            body = "".join(f"/{name} -> {upstream}\n" for name, upstream in sorted(self.config.routes.items()))
            self._send_text(HTTPStatus.OK, body, send_body=send_body)
            return

        route_name, separator, suffix = parsed.path.lstrip("/").partition("/")
        upstream = self.config.routes.get(route_name)
        if not separator or upstream is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown mirror route")
            return

        target = f"{upstream}/{suffix.lstrip('/')}"
        if parsed.query:
            target = f"{target}?{parsed.query}"

        if not self.config.slots.acquire(blocking=False):
            self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Proxy concurrency limit reached")
            return
        try:
            self._proxy(target, send_body=send_body, started=started)
        finally:
            self.config.slots.release()

    def _proxy(self, target: str, *, send_body: bool, started: float) -> None:
        headers = {name: value for name, value in self.headers.items() if name.lower() in REQUEST_HEADERS}
        headers.setdefault("User-Agent", "ReMe mirror proxy")
        request = urllib.request.Request(target, headers=headers, method=self.command)

        response = None
        try:
            # HTTPError is also a readable response and is handled by the same
            # cleanup block below, so a single with statement is not suitable.
            # pylint: disable=consider-using-with
            response = urllib.request.urlopen(  # noqa: S310 - targets are operator-configured
                request,
                timeout=self.config.timeout,
                context=ssl.create_default_context(),
            )
        except urllib.error.HTTPError as exc:
            response = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logging.error("upstream request failed target=%s error=%s", target, exc)
            self.send_error(
                HTTPStatus.BAD_GATEWAY,
                f"Upstream request failed: {exc.reason if isinstance(exc, urllib.error.URLError) else exc}",
            )
            return

        try:
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in RESPONSE_HEADERS_TO_SKIP:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()

            transferred = 0
            if send_body:
                while chunk := response.read(BUFFER_SIZE):
                    self.wfile.write(chunk)
                    transferred += len(chunk)
            elapsed = time.monotonic() - started
            logging.info(
                "%s %s -> %s status=%s bytes=%s elapsed=%.3fs",
                self.client_address[0],
                self.path,
                target,
                response.status,
                transferred,
                elapsed,
            )
        except (BrokenPipeError, ConnectionResetError):
            logging.warning("client disconnected path=%s target=%s", self.path, target)
        finally:
            response.close()

    def _send_text(self, status: HTTPStatus, body: str, *, send_body: bool) -> None:
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        if send_body:
            self.wfile.write(encoded)

    def log_message(self, format_string: str, *args: object) -> None:
        """Send the built-in access message to debug logging."""
        logging.debug("%s - %s", self.client_address[0], format_string % args)


class MirrorProxyServer(ThreadingHTTPServer):
    """Threading server carrying immutable proxy configuration."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], config: ProxyConfig):
        self.proxy_config = config
        super().__init__(address, MirrorProxyHandler)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default=DEFAULT_BIND, help=f"listen address (default: {DEFAULT_BIND})")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"listen port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        type=parse_route,
        metavar="NAME=URL",
        help="add or replace a route; may be repeated",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        type=parse_network,
        metavar="IP_OR_CIDR",
        help="allow a client address/network in addition to loopback; may be repeated",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="upstream timeout in seconds",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"maximum simultaneous upstream requests (default: {DEFAULT_MAX_CONCURRENCY})",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser


def main() -> int:
    """Run the mirror proxy until SIGINT or SIGTERM."""
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be at least 1")

    routes = dict(DEFAULT_ROUTES)
    routes.update(args.route)
    loopback_networks = [parse_network("127.0.0.0/8"), parse_network("::1/128")]
    config = ProxyConfig(
        routes=routes,
        allowed_clients=tuple(loopback_networks + args.allow),
        timeout=args.timeout,
        slots=threading.BoundedSemaphore(args.max_concurrency),
    )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    server = MirrorProxyServer((args.bind, args.port), config)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(
            signum,
            lambda _signum, _frame: threading.Thread(target=server.shutdown).start(),
        )

    logging.info("listening on http://%s:%s routes=%s", args.bind, args.port, sorted(routes))
    logging.info("allowed clients=%s", [str(network) for network in config.allowed_clients])
    try:
        server.serve_forever()
    finally:
        server.server_close()
        logging.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
