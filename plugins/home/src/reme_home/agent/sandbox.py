"""Home-specific Git runtime setup around ReMe's unchanged OS file sandboxes."""

import os
import subprocess
import sys
from pathlib import Path

from reme.utils.file_sandbox import create_file_sandbox_backend


def _developer_directory():
    """Locate Apple's actual Git runtime without granting its parent directories."""
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/bin/xcode-select", "-p"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            path = Path(result.stdout.strip()).resolve(strict=True)
            if (path / "usr/bin/git").is_file() and path != Path("/"):
                return str(path)
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError):
        pass
    return None


class CaseStudySandbox:
    """Delegate execution and cleanup to the OS sandbox, adding only child environment values."""

    os_name = "posix"

    def __init__(self, backend, environment):
        self.backend = backend
        self.environment = dict(environment)

    async def getcwd(self):
        """Preserve the backend's contained working directory."""
        return await self.backend.getcwd()

    async def exec_shell(self, command, *, cwd=None, timeout=None):
        """Set the environment inside isolation using argv, never shell interpolation."""
        if not command:
            raise ValueError("command must not be empty")
        return await self.backend.exec_shell(
            ["/usr/bin/env", *[f"{name}={value}" for name, value in self.environment.items()], *command],
            cwd=cwd,
            timeout=timeout,
        )


def create_case_study_sandbox(*, read_only_paths, **kwargs):
    """Explicitly grant Home's Git runtime while preserving core isolation and resource limits."""
    roots = list(read_only_paths)
    developer = _developer_directory()
    if developer and not any(Path(developer).is_relative_to(root) for root in roots):
        roots.append(developer)
    binary_paths = [str(Path(sys.prefix) / "bin"), str(Path(sys.base_prefix) / "bin")]
    if developer:
        binary_paths.append(str(Path(developer) / "usr/bin"))
    binary_paths.extend(["/usr/local/bin", "/usr/bin", "/bin"])
    environment = {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "PATH": os.pathsep.join(dict.fromkeys(binary_paths)),
    }
    if developer:
        environment["DEVELOPER_DIR"] = developer
    backend = create_file_sandbox_backend(read_only_paths=roots, **kwargs)
    return CaseStudySandbox(backend, environment)
