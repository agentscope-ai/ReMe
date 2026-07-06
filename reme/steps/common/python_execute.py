"""Execute Python code and return printed output."""

import asyncio
import os
import sys
import tempfile

from ..base_step import BaseStep
from ...components import R


@R.register("python_execute_step")
class PythonExecuteStep(BaseStep):
    """Execute Python code and return stdout.

    Inputs:
        code (str, required): Python source code. The final result should be printed.
        timeout (float, optional): execution timeout in seconds, default 10.
    """

    async def execute(self):
        assert self.context is not None
        code: str = self.context.get("code", "")
        timeout = self.context.get("timeout", 10)

        if not code:
            self.context.response.success = False
            self.context.response.answer = "Skipped: empty code"
            return self.context.response
        try:
            timeout_s = float(timeout)
        except (TypeError, ValueError):
            self.context.response.success = False
            self.context.response.answer = "Invalid timeout"
            return self.context.response
        if timeout_s <= 0:
            self.context.response.success = False
            self.context.response.answer = "Invalid timeout"
            return self.context.response

        self.logger.info(f"[{self.name}] python execute code:\n{code}")
        with tempfile.TemporaryDirectory(prefix="reme_python_execute_") as tmpdir:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
            }
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                code,
                cwd=tmpdir,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                self.logger.warning(f"[{self.name}] python execute timed out after {timeout_s:g}s code:\n{code}")
                self.context.response.success = False
                self.context.response.answer = f"Error: Python execution timed out after {timeout_s:g}s"
                self.context.response.metadata.update(
                    {
                        "code": code,
                        "timeout": timeout_s,
                        "returncode": None,
                        "stdout": "",
                        "stderr": "timeout",
                    },
                )
                return self.context.response

        stdout = stdout_b.decode("utf-8", errors="replace").strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        success = process.returncode == 0
        answer = stdout if success else f"Error: Python exited with code {process.returncode}"
        if not success and stderr:
            answer = f"{answer}\n{stderr}"

        self.logger.info(
            f"[{self.name}] python execute returncode={process.returncode} stdout={stdout!r} stderr={stderr!r}",
        )
        self.context.response.success = success
        self.context.response.answer = answer
        self.context.response.metadata.update(
            {
                "code": code,
                "timeout": timeout_s,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )
        return self.context.response
