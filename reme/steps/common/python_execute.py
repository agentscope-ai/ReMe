"""Execute Python code and return printed stdout."""

import asyncio
import subprocess
import sys
from typing import Any

from ..base_step import BaseStep
from ...components import R

DEFAULT_TIMEOUT = 60.0


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

        try:
            result = await asyncio.to_thread(self._run_python, code, timeout)
        except subprocess.TimeoutExpired:
            self.context.response.success = False
            self.context.response.answer = f"Python execution timed out after {timeout:g}s"
            self.context.response.metadata.update({"timeout": timeout})
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

    def _run_python(self, code: str, timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.workspace_path,
            check=False,
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
