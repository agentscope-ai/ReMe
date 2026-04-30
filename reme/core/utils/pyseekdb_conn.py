"""Build ``pyseekdb.Client`` / ``AdminClient`` kwargs for embedded vs remote OceanBase / seekdb."""

from __future__ import annotations

DEFAULT_SEEKDB_PORT = 2881
DEFAULT_SEEKDB_TENANT = "sys"
DEFAULT_SEEKDB_USER = "root"


def parse_host_port(host: str | None, port: int | None) -> tuple[str | None, int | None]:
    """Resolve remote ``host`` / ``port`` (``port`` defaults to :data:`DEFAULT_SEEKDB_PORT`)."""
    if host and host.strip():
        return host.strip(), port if port is not None else DEFAULT_SEEKDB_PORT
    return None, None


def build_pyseekdb_client_kwargs(
    *,
    path: str | None = None,
    database: str,
    host: str | None = None,
    port: int | None = None,
    tenant: str = DEFAULT_SEEKDB_TENANT,
    user: str | None = None,
    password: str = "",
) -> tuple[bool, dict]:
    """Return ``(is_remote, kwargs)`` for ``pyseekdb.Client``.

    Remote when ``host`` is set. Embedded: optional ``path`` for the data directory; if
    omitted, pyseekdb uses its default (typically ``seekdb.db`` under the CWD).
    """
    h, p = parse_host_port(host, port)
    if h:
        return True, {
            "host": h,
            "port": p,
            "tenant": tenant,
            "database": database,
            "user": user if user is not None else DEFAULT_SEEKDB_USER,
            "password": password,
        }
    kw: dict = {"database": database}
    if path:
        kw["path"] = path
    return False, kw


def admin_kwargs_from_client_kwargs(client_kw: dict) -> dict:
    """Strip ``database`` for ``AdminClient`` (admin uses system DB)."""
    if "path" in client_kw:
        return {"path": client_kw["path"]}
    if "host" in client_kw:
        return {
            "host": client_kw["host"],
            "port": client_kw["port"],
            "tenant": client_kw["tenant"],
            "user": client_kw["user"],
            "password": client_kw["password"],
        }
    return {}
