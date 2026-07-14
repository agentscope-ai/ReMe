"""Execute a shell command and return its stdout."""

import asyncio
from dataclasses import dataclass
from typing import Any

from ..base_step import BaseStep
from ...components import R

DEFAULT_TIMEOUT = 60.0 * 60 * 24


@dataclass(frozen=True)
class _ShellResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False


@R.register("shell_step")
class ShellStep(BaseStep):
    """Run a command in a shell and return stdout as the response answer."""

    async def execute(self):
        assert self.context is not None

        command = self.context.get("cmd", self.context.get("command", ""))
        raw_timeout = self.context.get("shell_timeout", self.context.get("timeout", DEFAULT_TIMEOUT))
        timeout, timeout_error = self._parse_timeout(raw_timeout)
        if not isinstance(command, str) or not command.strip():
            self.context.response.success = False
            self.context.response.answer = "cmd is required"
            return self.context.response
        if timeout_error:
            self.context.response.success = False
            self.context.response.answer = timeout_error
            return self.context.response

        result = await self._run_shell(command, timeout)
        if result.timed_out:
            self.context.response.success = False
            self.context.response.answer = f"Shell command timed out after {timeout:g}s"
        else:
            self.context.response.success = result.returncode == 0
            self.context.response.answer = result.stdout if result.stdout or result.returncode == 0 else result.stderr

        self.context.response.metadata.update(
            {
                "returncode": result.returncode,
                "stderr": result.stderr,
                "timeout": timeout,
            },
        )
        return self.context.response

    async def _run_shell(self, command: str, timeout: float) -> _ShellResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace_path,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return _ShellResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                returncode=process.returncode,
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return _ShellResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                returncode=process.returncode,
                timed_out=True,
            )

    @staticmethod
    def _parse_timeout(raw: Any) -> tuple[float, str]:
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT, "timeout must be a positive number"
        if timeout <= 0:
            return DEFAULT_TIMEOUT, "timeout must be a positive number"
        return timeout, ""
