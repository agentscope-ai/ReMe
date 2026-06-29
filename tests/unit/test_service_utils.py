"""Tests for the psutil-based service discovery helpers in ``service_utils``.

Covers the cross-platform replacements for the old ``lsof`` / ``pgrep``
shell-outs:

* ``_pid_on_port`` — real integration (open a listening socket, find its
  PID) plus a mock test proving inaccessible processes are skipped, not
  crashed on.
* ``_scan_reme_procs`` — mock ``psutil.process_iter`` to assert cmdline
  parsing, defaulting, filtering, and AccessDenied tolerance.
"""

# pylint: disable=protected-access,missing-function-docstring,unused-argument

import os
import socket
from types import SimpleNamespace

import psutil

from reme.utils import service_utils as su


# ----------------------------------------------------------------------
# Fakes for mocking psutil.process_iter
# ----------------------------------------------------------------------


def _conn(port: int, status=psutil.CONN_LISTEN, has_laddr: bool = True):
    return SimpleNamespace(status=status, laddr=SimpleNamespace(port=port) if has_laddr else None)


class _FakeProc:
    """Stand-in for a psutil.Process yielded by process_iter."""

    def __init__(self, pid: int, *, cmdline=None, conns=None, conn_exc=None):
        self.info = {"pid": pid, "cmdline": cmdline if cmdline is not None else []}
        self._conns = conns or []
        self._conn_exc = conn_exc

    def net_connections(self, kind="tcp"):
        if self._conn_exc is not None:
            raise self._conn_exc
        return self._conns


class _RaisingInfoProc:
    """Process whose `.info` access raises (simulates AccessDenied in iteration)."""

    def __init__(self, exc):
        self._exc = exc

    @property
    def info(self):
        raise self._exc


def _patch_iter(monkeypatch, procs):
    monkeypatch.setattr(su.psutil, "process_iter", lambda attrs=None: iter(procs))


# ----------------------------------------------------------------------
# _pid_on_port
# ----------------------------------------------------------------------


def test_pid_on_port_finds_own_listening_socket():
    """Real integration: a listening socket is attributed to this process's PID."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert su._pid_on_port(port) == os.getpid()
    finally:
        srv.close()


def test_pid_on_port_none_when_nobody_listening():
    """A port with no listener resolves to None."""
    # Grab then immediately release an ephemeral port to get a number that is
    # (almost certainly) unbound right now.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()
    assert su._pid_on_port(free_port) is None


def test_pid_on_port_skips_inaccessible_processes(monkeypatch):
    """AccessDenied on one process must not abort the scan of the rest."""
    procs = [
        _FakeProc(111, conn_exc=psutil.AccessDenied(pid=111)),
        _FakeProc(222, conn_exc=psutil.NoSuchProcess(pid=222)),
        _FakeProc(333, conns=[_conn(9999)]),
    ]
    _patch_iter(monkeypatch, procs)
    assert su._pid_on_port(9999) == 333


def test_pid_on_port_ignores_non_listen_and_mismatched(monkeypatch):
    """Only LISTEN sockets on the exact port match."""
    procs = [
        _FakeProc(1, conns=[_conn(9999, status=psutil.CONN_ESTABLISHED)]),  # right port, wrong state
        _FakeProc(2, conns=[_conn(8888)]),  # listening, wrong port
        _FakeProc(3, conns=[_conn(9999, has_laddr=False)]),  # no laddr
    ]
    _patch_iter(monkeypatch, procs)
    assert su._pid_on_port(9999) is None


# ----------------------------------------------------------------------
# _scan_reme_procs
# ----------------------------------------------------------------------


def test_scan_reme_procs_parses_host_and_port(monkeypatch):
    procs = [
        _FakeProc(
            123,
            cmdline=["python", "-m", "reme.reme", "start", "service.host=0.0.0.0", "service.port=8123"],
        ),
    ]
    _patch_iter(monkeypatch, procs)
    assert su._scan_reme_procs() == [(123, "0.0.0.0", 8123)]


def test_scan_reme_procs_defaults_when_args_absent(monkeypatch):
    procs = [_FakeProc(7, cmdline=["reme", "start"])]
    _patch_iter(monkeypatch, procs)
    assert su._scan_reme_procs() == [(7, su.REME_DEFAULT_HOST, su.REME_DEFAULT_PORT)]


def test_scan_reme_procs_filters_unrelated(monkeypatch):
    procs = [
        _FakeProc(1, cmdline=["reme", "find_reme"]),  # no 'start' token
        _FakeProc(2, cmdline=["python", "-m", "http.server", "start"]),  # no reme token
        _FakeProc(3, cmdline=[]),  # empty cmdline
    ]
    _patch_iter(monkeypatch, procs)
    assert not su._scan_reme_procs()


def test_scan_reme_procs_ignores_non_digit_port(monkeypatch):
    """A malformed service.port= falls back to the default port, not a crash."""
    procs = [_FakeProc(9, cmdline=["reme", "start", "service.port=notaport"])]
    _patch_iter(monkeypatch, procs)
    assert su._scan_reme_procs() == [(9, su.REME_DEFAULT_HOST, su.REME_DEFAULT_PORT)]


def test_scan_reme_procs_skips_access_denied(monkeypatch):
    """A process that denies cmdline access is skipped, others still scanned."""
    procs = [
        _RaisingInfoProc(psutil.AccessDenied(pid=99)),
        _FakeProc(5, cmdline=["reme", "start"]),
    ]
    _patch_iter(monkeypatch, procs)
    assert su._scan_reme_procs() == [(5, su.REME_DEFAULT_HOST, su.REME_DEFAULT_PORT)]
