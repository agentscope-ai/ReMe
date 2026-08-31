"""Execute Python code and return printed stdout."""

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

from ..base_step import BaseStep
from ...components import R

DEFAULT_TIMEOUT = 60.0


@dataclass(frozen=True)
class _PythonResult:
    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False


@R.register("python_execute_step")
class PythonExecuteStep(BaseStep):
    """Run Python code in a subprocess and return stdout as the response answer."""

    async def execute(self):
        assert self.context is not None

        code = self.context.get("code", "")
        timeout, timeout_error = self._parse_timeout(self.context.get("timeout", DEFAULT_TIMEOUT))
        if not isinstance(code, str) or not code.strip():
            self.context.response.success = False
            self.context.response.answer = "code is required"
            return self.context.response
        if timeout_error:
            self.context.response.success = False
            self.context.response.answer = timeout_error
            return self.context.response

        result = await self._run_python(code, timeout)
        if result.timed_out:
            self.context.response.success = False
            self.context.response.answer = f"Python execution timed out after {timeout:g}s"
            self.context.response.metadata.update(
                {
                    "returncode": result.returncode,
                    "stderr": result.stderr,
                    "timeout": timeout,
                },
            )
            return self.context.response

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        self.context.response.success = result.returncode == 0
        self.context.response.answer = stdout if stdout or result.returncode == 0 else stderr
        self.context.response.metadata.update(
            {
                "returncode": result.returncode,
                "stderr": stderr,
                "timeout": timeout,
            },
        )
        return self.context.response

    async def _run_python(self, code: str, timeout: float) -> _PythonResult:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.workspace_path,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return _PythonResult(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
                returncode=process.returncode,
            )
        except TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return _PythonResult(
                stdout=stdout.decode(),
                stderr=stderr.decode(),
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
