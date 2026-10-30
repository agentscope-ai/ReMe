"""Execute Python code and return printed stdout."""

import sys
from typing import Any

from ...utils.local_backend import LocalProcessBackend, ProcessTimeout
from ..base_step import BaseStep
from ...components import R
from .file_sandbox import file_sandbox_for_step

DEFAULT_TIMEOUT = 60.0


@R.register("python_execute_step")
class PythonExecuteStep(BaseStep):
    """Run Python code and return stdout, limiting each output stream to 1 MiB."""

    async def execute(self):
        assert self.context is not None

        code = self.context.get("code", "")
        timeout, timeout_error = self._parse_timeout(
            self.context.get("python_timeout", self.context.get("timeout", DEFAULT_TIMEOUT)),
        )
        if not isinstance(code, str) or not code.strip():
            self.context.response.success = False
            self.context.response.answer = "code is required"
            return self.context.response
        if timeout_error:
            self.context.response.success = False
            self.context.response.answer = timeout_error
            return self.context.response

        timed_out = False
        try:
            backend = file_sandbox_for_step(self)
            result = await (backend or LocalProcessBackend()).exec_shell(
                [sys.executable, "-c", code],
                cwd=str(self.workspace_path),
                timeout=timeout,
                **({"raise_errors": True} if backend is None else {}),
            )
        except ProcessTimeout as exc:
            result, timed_out = exc.result, True
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response
        timed_out = timed_out or (result.exit_code == -1 and result.stderr == b"timed out")
        stdout = result.stdout.decode(errors="replace")
        stderr = result.stderr.decode(errors="replace")
        if timed_out:
            self.context.response.success = False
            self.context.response.answer = f"Python execution timed out after {timeout:g}s"
        else:
            self.context.response.success = result.exit_code == 0
            self.context.response.answer = stdout if stdout or result.exit_code == 0 else stderr
        self.context.response.metadata.update(
            {
                "returncode": result.exit_code,
                "stderr": stderr,
                "python_timeout": timeout,
            },
        )
        return self.context.response

    @staticmethod
    def _parse_timeout(raw: Any) -> tuple[float, str]:
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT, "python_timeout must be a positive number"
        if timeout <= 0:
            return DEFAULT_TIMEOUT, "python_timeout must be a positive number"
        return timeout, ""
