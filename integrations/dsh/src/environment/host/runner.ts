import { spawn } from "node:child_process";

export interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
  /** Spawn or timeout failure description; absent on a normal exit. */
  error?: string;
}

export interface RunOptions {
  timeoutMs?: number;
  cwd?: string;
  /** Join argv verbatim (cmd.exe /s /c wrappers); never combine with user input. */
  windowsVerbatimArguments?: boolean;
}

export type RunCommand = (
  exe: string,
  args: string[],
  options?: RunOptions,
) => Promise<RunResult>;

/** UTF-8 guards so probe output survives GBK console codepages on Windows. */
export function probeEnv(base: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
  return {
    ...base,
    PYTHONIOENCODING: "utf-8",
    PYTHONUTF8: "1",
  };
}

/** Array-argv, shell-free runner with a hard timeout and no thrown rejections. */
export function createNodeRunner(
  base: NodeJS.ProcessEnv = process.env,
): RunCommand {
  return (exe, args, options = {}) =>
    new Promise((resolve) => {
      let child: ReturnType<typeof spawn>;
      try {
        child = spawn(exe, args, {
          shell: false,
          windowsHide: true,
          ...(options.windowsVerbatimArguments
            ? { windowsVerbatimArguments: true }
            : {}),
          ...(options.cwd ? { cwd: options.cwd } : {}),
          env: probeEnv(base),
        });
      } catch (error) {
        resolve({ code: null, stdout: "", stderr: "", error: String(error) });
        return;
      }
      let stdout = "";
      let stderr = "";
      let settled = false;
      let timedOut = false;
      const timers: ReturnType<typeof setTimeout>[] = [];
      const settle = (result: RunResult): void => {
        if (settled) return;
        settled = true;
        for (const timer of timers) clearTimeout(timer);
        resolve(result);
      };
      const timeoutMs = options.timeoutMs ?? 10_000;
      const timer = setTimeout(() => {
        // The caller must see an explicit error: an exit caused by our own
        // timeout is not "the interpreter has no reme-ai".
        timedOut = true;
        child.kill();
        // A child that ignores SIGTERM must not make the probe hang forever:
        // force it and settle, rather than waiting for a close that may never
        // arrive.
        const grace = setTimeout(() => {
          try {
            child.kill("SIGKILL");
          } catch {
            // Already exited.
          }
          settle({
            code: null,
            stdout,
            stderr,
            error: `Command timed out after ${timeoutMs} ms`,
          });
        }, 1_000);
        grace.unref();
        timers.push(grace);
      }, timeoutMs);
      timer.unref();
      timers.push(timer);
      child.stdout?.on("data", (chunk) => {
        stdout += String(chunk);
      });
      child.stderr?.on("data", (chunk) => {
        stderr += String(chunk);
      });
      child.on("error", (error) => {
        settle({ code: null, stdout, stderr, error: String(error) });
      });
      child.on("close", (code) => {
        settle({
          code,
          stdout,
          stderr,
          ...(timedOut
            ? { error: `Command timed out after ${timeoutMs} ms` }
            : {}),
        });
      });
    });
}
