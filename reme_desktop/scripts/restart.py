"""Restart this checkout's macOS backend and desktop without changing workspace data."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, requires, version
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[2]
DESKTOP = ROOT / "reme_desktop"
WORKSPACE = ROOT / ".reme"
RUN_DIR = DESKTOP / "reports" / "run"
PORT = 2333
ENDPOINT = f"http://127.0.0.1:{PORT}"
ELECTRON = DESKTOP / "node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"

sys.path.insert(0, str(ROOT))


def check_environment():
    """Check the selected config in a separate process without starting jobs or writing workspace data."""
    if sys.version_info < (3, 11):
        raise RuntimeError("ReMe 需要 Python 3.11 或更新版本。")
    from reme.config import resolve_app_config
    from reme.utils.env_utils import load_env

    load_env(ROOT / ".env")
    config = resolve_app_config(config="proactive", log_config=False)
    from packaging.requirements import Requirement

    # Validate this checkout's declared runtime requirements, including enabled
    # extras. A successful import alone does not establish SDK compatibility.
    extras = {"core", "as", "finance", "web"}
    for raw in requires("reme-ai") or []:
        requirement = Requirement(raw)
        if requirement.name in {"reme-ai", "reme_studio"}:
            continue
        if requirement.marker and not any(requirement.marker.evaluate({"extra": extra}) for extra in extras):
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"缺少依赖 {requirement}；请安装当前仓库的 .[dev,core]。") from exc
        if installed not in requirement.specifier:
            raise RuntimeError(f"依赖版本不匹配：{requirement.name}=={installed}，需要 {requirement.specifier}。")
    for job in config.get("jobs", {}).values():
        if job.get("enabled", True) and any(
            step.get("backend") in {"screen_capture_step", "work_observation_step"} for step in job.get("steps", [])
        ):
            import_module("mss")
    from openai_codex import AsyncCodex

    if "include_turns" not in inspect.signature(AsyncCodex.thread_fork).parameters:
        raise RuntimeError("当前 openai-codex 不支持提醒 fork 协议，请安装当前仓库的 .[dev,core]。")
    stock_jobs = [
        name
        for name, job in config.get("jobs", {}).items()
        if job.get("enabled", True) and any(step.get("backend") == "stock_quotes_step" for step in job.get("steps", []))
    ]
    if stock_jobs:
        try:
            import_module("exchange_calendars")
        except ImportError as exc:
            raise RuntimeError(
                f"已启用 {', '.join(stock_jobs)}，但此 Python 缺少可用的 exchange-calendars；"
                "请安装 reme-ai[finance] 或选择完整的 ReMe 环境。",
            ) from exc


def owns_backend(process):
    """A port alone is insufficient evidence that a process belongs to this checkout."""
    from reme.utils.service_utils import _start_arguments

    return Path(process.cwd()).resolve() == ROOT and _start_arguments(process.cmdline()) is not None


def owns_desktop(process):
    """Recognize only this checkout's packaged app or its source Electron entry point."""
    arguments = process.cmdline()
    if not arguments:
        return False
    binary = Path(arguments[0]).resolve()
    packaged = [
        DESKTOP / f"dist/{directory}/ReMe Desktop.app/Contents/MacOS/ReMe Desktop" for directory in ("mac", "mac-arm64")
    ]
    if binary in packaged:
        return True
    if binary != ELECTRON or Path(process.cwd()).resolve() != DESKTOP:
        return False
    return any(not argument.startswith("-") and (DESKTOP / argument).resolve() == DESKTOP for argument in arguments[1:])


def listeners():
    """Use macOS lsof without requesting system-wide privileged connection access."""
    result = subprocess.run(
        ["lsof", "-nP", f"-tiTCP:{PORT}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("无法检查后端端口。")
    return [psutil.Process(int(pid)) for pid in set(result.stdout.split())]


def stop(process):
    """Allow normal shutdown and refuse to force-kill a process that has not finished."""
    try:
        process.terminate()
        process.wait(timeout=20)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired as exc:
        raise RuntimeError(f"进程 {process.pid} 尚未退出，重启已停止。") from exc


def spawn(arguments, directory, log_name, **kwargs):
    """Detach services from Terminal while retaining their startup and runtime logs."""
    with (RUN_DIR / log_name).open("ab", buffering=0) as output:
        return subprocess.Popen(
            arguments,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **kwargs,
        )


def wait_backend(process):
    """Read AgentRuntime readiness rather than assuming a listening socket is sufficient."""
    with httpx.Client(base_url=ENDPOINT, timeout=2, trust_env=False) as client:
        deadline = time.monotonic() + 60
        while process.poll() is None and time.monotonic() < deadline:
            try:
                value = client.post("/agent_list", json={}).json()
                if value.get("success") and any(item["agent_id"] == "chat" for item in value["answer"]["items"]):
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.25)
    raise RuntimeError(f"后端启动失败，请查看 {RUN_DIR / 'backend.log'}")


def wait_desktop(process):
    """Confirm the actual desktop process established its backend connection."""
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        try:
            if any(
                conn.status == psutil.CONN_ESTABLISHED and conn.raddr and conn.raddr.port == PORT
                for conn in psutil.Process(process.pid).net_connections(kind="tcp")
            ):
                return
        except psutil.NoSuchProcess:
            break
        time.sleep(0.25)
    raise RuntimeError(f"桌面端未连接后端，请查看 {RUN_DIR / 'desktop.log'}")


def restart():
    """Preflight, build, stop owned processes, then start and verify both services."""
    if sys.platform != "darwin":
        raise RuntimeError("此一键脚本用于 macOS。")
    if not ELECTRON.is_file():
        raise RuntimeError("请先在 reme_desktop 中运行 npm ci 和 npx install-electron。")
    backend_processes = listeners()
    if any(not owns_backend(process) for process in backend_processes):
        raise RuntimeError(f"端口 {PORT} 被其他程序占用，未停止任何进程。")

    # Keep .env credentials inside the check/backend processes, outside Desktop's environment.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--check"], cwd=ROOT, check=True)
    print("[1/3] 构建最新桌面端……", flush=True)
    subprocess.run(["npm", "run", "build"], cwd=DESKTOP, check=True)
    print("[2/3] 停止当前仓库的桌面端和后端……", flush=True)
    for process in psutil.process_iter():
        try:
            if owns_desktop(process):
                stop(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    for process in backend_processes:
        stop(process)

    print("[3/3] 启动后端和桌面端……", flush=True)
    backend = spawn(
        [
            sys.executable,
            "-m",
            "reme.reme",
            "start",
            "config=proactive",
            f"workspace_dir={WORKSPACE}",
            "service.host=127.0.0.1",
            f"service.port={PORT}",
            "log_config=false",
            "log_to_file=false",
        ],
        ROOT,
        "backend.log",
    )
    desktop = None
    try:
        wait_backend(backend)
        environment = {**os.environ, "REME_DESKTOP_API_URL": ENDPOINT}
        desktop = spawn([str(ELECTRON), str(DESKTOP)], DESKTOP, "desktop.log", env=environment)
        wait_desktop(desktop)
        # The existing single-instance handler opens the conversation window.
        with (RUN_DIR / "desktop.log").open("ab") as output:
            subprocess.run(
                [str(ELECTRON), str(DESKTOP)],
                cwd=DESKTOP,
                env=environment,
                timeout=10,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except Exception:
        for process in (desktop, backend):
            if process is not None and process.poll() is None:
                stop(psutil.Process(process.pid))
        raise
    print(f"已重启并连接：{ENDPOINT}\n工作区：{WORKSPACE}\n日志：{RUN_DIR}", flush=True)


def main():
    """Serialize double-clicks so two restarts cannot stop each other's new services."""
    if sys.argv[1:] == ["--check"]:
        try:
            check_environment()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"Python 环境检查失败（{sys.executable}）：{exc}", file=sys.stderr)
            return 1
        return 0
    if sys.platform != "darwin":
        print("此一键脚本用于 macOS。", file=sys.stderr)
        return 1
    import fcntl

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with (RUN_DIR / "restart.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            restart()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            print(f"重启失败：{exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
