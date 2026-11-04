"""Execute Python code and return printed stdout."""

import math
import sys
from typing import Any

from ...utils.local_backend import ProcessTimeout
from ..base_step import BaseStep
from ...components import R
from .file_sandbox import file_sandbox_for_step

DEFAULT_TIMEOUT = 60.0
MAX_TIMEOUT = 86400.0


@R.register("python_execute_step")
class PythonExecuteStep(BaseStep):
    """Run sandboxed Python with a read-only workspace and bounded output.

    Timeouts default to 60 seconds and are capped at one day. A trusted Step
    configuration can set max_python_timeout to a shorter limit; request data
    cannot override that limit. Positive infinity is capped; NaN is rejected.
    """

    async def execute(self):
        assert self.context is not None

        code = self.context.get("code", "")
        maximum, maximum_error = self._parse_timeout(self.kwargs.get("max_python_timeout", MAX_TIMEOUT))
        timeout, timeout_error = self._parse_timeout(
            self.context.get("python_timeout", self.context.get("timeout", DEFAULT_TIMEOUT)),
            maximum=maximum,
        )
        if maximum_error:
            timeout_error = maximum_error.replace("python_timeout", "max_python_timeout")
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
            result = await backend.exec_shell(
                [sys.executable, "-c", code],
                cwd=str(self.workspace_path.resolve()),
                timeout=timeout,
                raise_errors=True,
            )
        except ProcessTimeout as exc:
            result, timed_out = exc.result, True
        except (TypeError, ValueError, OSError, RuntimeError) as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response
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
    def _parse_timeout(raw: Any, *, maximum: float = MAX_TIMEOUT) -> tuple[float, str]:
        try:
            if isinstance(raw, bool):
                raise ValueError("boolean timeout")
            timeout = float(raw)
        except OverflowError:
            # JSON integers can exceed float's range. They still have a known
            # sign, so large positive integers follow the normal cap policy.
            if isinstance(raw, int) and raw > 0:
                return maximum, ""
            return DEFAULT_TIMEOUT, "python_timeout must be a positive number"
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT, "python_timeout must be a positive number"
        if math.isnan(timeout) or timeout <= 0:
            return DEFAULT_TIMEOUT, "python_timeout must be a positive number"
        return min(timeout, maximum), ""
