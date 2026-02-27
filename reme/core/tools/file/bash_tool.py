"""Bash command execution tool with production-grade features.

This module provides a production-grade tool for executing bash commands with:
- Smart output truncation (keeps last N lines/bytes to prevent memory issues)
- Process tree termination (prevents orphan processes)
"""

import asyncio
import os
import platform
import signal
from pathlib import Path

from .base_file_tool import BaseFileTool
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_tail
from ...schema import ToolCall, TruncationResult


def get_shell_config() -> tuple[str, list[str]]:
    """Get the appropriate shell and arguments for the current platform.

    Returns:
        Tuple of (shell_path, args) for subprocess execution
    """
    system = platform.system()

    if system == "Windows":
        # Use PowerShell on Windows
        return "powershell.exe", ["-Command"]
    else:
        # Use bash on Unix-like systems
        shell = os.environ.get("SHELL", "/bin/bash")
        return shell, ["-c"]


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children.

    Args:
        pid: Process ID to kill
    """
    try:
        if platform.system() == "Windows":
            # Windows: use taskkill
            os.system(f"taskkill /F /T /PID {pid}")
        else:
            # Unix: kill process group
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass  # Process already dead
    except Exception:
        pass  # Best effort


class BashTool(BaseFileTool):
    """Production-grade tool for executing bash commands.

    Features:
    - Smart output truncation (preserves last N lines or M bytes)
    - Kills entire process tree on timeout (prevents orphan processes)
    """

    def __init__(self, cwd: str | None = None, command_prefix: str | None = None):
        """Initialize bash tool.

        Args:
            cwd: Working directory (defaults to current directory)
            command_prefix: Optional prefix prepended to every command
        """
        super().__init__()
        self.cwd = cwd or os.getcwd()
        self.command_prefix = command_prefix

    def _build_tool_call(self) -> ToolCall:
        max_kb = DEFAULT_MAX_BYTES // 1024
        return ToolCall(
            **{
                "description": (
                    f"Execute a bash command in the current working directory. "
                    f"Returns stdout and stderr. Output is truncated to last "
                    f"{DEFAULT_MAX_LINES} lines or {max_kb}KB (whichever is hit first). "
                    f"Optionally provide a timeout in seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Bash command to execute",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (optional, no default timeout)",
                        },
                    },
                    "required": ["command"],
                },
            },
        )

    async def execute(self) -> str:
        """Execute the bash command with production-grade features."""
        command: str = self.context.command
        timeout: float | None = self.context.get("timeout", None)

        # Apply command prefix if configured
        if self.command_prefix:
            command = f"{self.command_prefix}\n{command}"

        # Verify working directory exists
        if not Path(self.cwd).exists():
            raise FileNotFoundError(
                f"Working directory does not exist: {self.cwd}\n" f"Cannot execute bash commands.",
            )

        # Get shell configuration
        shell, shell_args = get_shell_config()

        # Start process
        process = await asyncio.create_subprocess_exec(
            shell,
            *shell_args,
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            # Create process group for clean termination
            preexec_fn=os.setpgrp if platform.system() != "Windows" else None,
        )

        # Execute command with optional timeout
        if timeout and timeout > 0:
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as e:
                # Kill process tree on timeout
                if process.pid:
                    kill_process_tree(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    process.kill()
                raise TimeoutError(f"Command timed out after {timeout} seconds") from e
        else:
            stdout, stderr = await process.communicate()

        # Decode output
        full_output = stdout.decode("utf-8", errors="ignore")
        if stderr:
            stderr_text = stderr.decode("utf-8", errors="ignore")
            if full_output:
                full_output += "\n"
            full_output += stderr_text

        # Apply tail truncation_result to prevent memory issues
        truncation_result: TruncationResult = truncate_tail(full_output)
        output_text = truncation_result.content or "(no output)"

        # Build truncation_result notice if needed
        if truncation_result.truncated:
            start_line = truncation_result.total_lines - truncation_result.output_lines + 1
            end_line = truncation_result.total_lines

            if truncation_result.truncated_by == "lines":
                output_text += (
                    f"\n\n[Output truncated: showing lines {start_line}-{end_line} "
                    f"of {truncation_result.total_lines} total lines]"
                )
            else:
                max_kb = DEFAULT_MAX_BYTES // 1024
                output_text += (
                    f"\n\n[Output truncated: showing lines {start_line}-{end_line} "
                    f"of {truncation_result.total_lines} ({max_kb}KB limit reached)]"
                )

        # Handle non-zero exit code
        if process.returncode != 0:
            output_text += f"\n\nCommand exited with code {process.returncode}"
            raise RuntimeError(output_text)

        return output_text
