import { spawn } from "node:child_process";
import { closeSync, fstatSync, mkdirSync, openSync, readSync } from "node:fs";
import { join } from "node:path";

import type { InstallPhase, InstallRequest, InstallStatus } from "../types.js";
import { interpreterKey } from "../interpreter-key.js";
import { INSTALL_LOG_KEEP, openAppendLog, pruneInstallLogs } from "./log.js";
import { isAbsolutePath } from "./paths.js";

export interface InstallChild {
  pid: number | undefined;
  onExit(callback: (code: number | null) => void): void;
  /** `force` escalates to SIGKILL on POSIX; Windows always force-kills. */
  kill(force?: boolean): void;
}

export interface InstallSpawner {
  start(options: {
    exe: string;
    args: string[];
    logPath: string;
  }): InstallChild;
}

/** Compose the pip invocation for `reme-ai[core]`. */
export function pipArgs(request: InstallRequest): string[] {
  const args = [
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
    "--no-input",
  ];
  if (request.upgrade) args.push("-U");
  // `pip install reme-ai[core]` reports "already satisfied" for a broken but
  // present install, so an explicit Reinstall must force it.
  if (request.forceReinstall) args.push("--force-reinstall");
  const indexUrl = (request.indexUrl ?? "").trim();
  if (indexUrl) {
    if (!/^https?:\/\/\S+$/i.test(indexUrl))
      throw new TypeError("Package index must be an http(s) URL");
    // Credentials embedded in the URL would be echoed in the job command, the
    // install log and the browser; pip's own config is the safe place for them.
    const parsed = new URL(indexUrl);
    if (parsed.username || parsed.password)
      throw new TypeError(
        "Package index must not embed credentials; configure them in pip.conf",
      );
    args.push("--index-url", indexUrl);
  }
  args.push("reme-ai[core]");
  return args;
}

/**
 * Last `maxLines` non-empty lines of a log file, normalized to "\n". Only the
 * tail is read: this runs on every 2 s poll and the log can grow for minutes.
 */
export function tailLog(path: string, maxLines = 30, maxBytes = 8192): string {
  let fd: number | undefined;
  try {
    fd = openSync(path, "r");
    const size = fstatSync(fd).size;
    const length = Math.min(size, maxBytes);
    const start = size - length;
    const buffer = Buffer.alloc(length);
    readSync(fd, buffer, 0, length, start);
    let text = buffer.toString("utf8");
    if (start > 0) {
      // The cut may be mid-line and even mid-character; drop that partial line
      // so the tail never starts with a replacement character.
      const firstBreak = text.indexOf("\n");
      text =
        firstBreak >= 0
          ? text.slice(firstBreak + 1)
          : text.replace(/^\uFFFD+/, "");
    }
    text = text.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
    return text
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .slice(-maxLines)
      .join("\n");
  } catch {
    return "";
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // Already closed.
      }
    }
  }
}

/** Spawn pip with stdout/stderr appended to a per-job log file. */
export function createInstallSpawner(
  base: NodeJS.ProcessEnv = process.env,
): InstallSpawner {
  return {
    start({ exe, args, logPath }) {
      const fd = openAppendLog(logPath);
      let child: ReturnType<typeof spawn>;
      try {
        child = spawn(exe, args, {
          shell: false,
          windowsHide: true,
          // POSIX: lead a process group so a cancel can stop pip's build
          // backends too, not only pip itself.
          detached: process.platform !== "win32",
          env: {
            ...base,
            PYTHONIOENCODING: "utf-8",
            PYTHONUTF8: "1",
            PYTHONUNBUFFERED: "1",
          },
          stdio: ["ignore", fd, fd],
        });
      } finally {
        // The child owns its inherited handle; the parent copy is no longer needed.
        closeSync(fd);
      }
      return {
        pid: child.pid,
        onExit(callback) {
          let settled = false;
          const finish = (code: number | null): void => {
            if (settled) return;
            settled = true;
            callback(code);
          };
          child.on("close", finish);
          child.on("error", () => finish(null));
        },
        kill(force = false) {
          if (child.pid !== undefined && process.platform === "win32") {
            // Tree-kill: pip's build backends are child processes.
            const killer = spawn(
              "taskkill",
              ["/PID", String(child.pid), "/T", "/F"],
              { windowsHide: true, stdio: "ignore" },
            );
            // Without this listener an async spawn failure would exit the Host.
            killer.on("error", () => undefined);
            killer.unref();
            return;
          }
          const signal = force ? "SIGKILL" : "SIGTERM";
          // The install process leads its own group on POSIX, so pip's build
          // backends receive the signal as well.
          try {
            if (child.pid !== undefined) process.kill(-child.pid, signal);
            return;
          } catch {
            // Not a group leader (e.g. spawn failed); signal the process only.
          }
          try {
            child.kill(signal);
          } catch {
            // Already gone.
          }
        },
      };
    },
  };
}

/**
 * Whether two interpreter paths name the same interpreter.
 *
 * The path flavor decides the comparison: Windows paths fold case and
 * separators, POSIX paths do not, because they are case-sensitive there. The
 * tab compares the same way, so a job it found is never dropped again.
 */
function sameInterpreter(left: string, right: string): boolean {
  return interpreterKey(left) === interpreterKey(right);
}

/**
 * Default grace period between the graceful termination signal and the
 * force-kill of a cancelled job. pip gets a moment to clean up its build
 * children before it is killed outright.
 */
const DEFAULT_KILL_GRACE_MS = 5_000;

/**
 * Default grace period for a disposed job.
 *
 * Zero: the plugin is unloading, so the force-kill follows the graceful signal
 * immediately and a child can never outlive the Host.
 */
const DEFAULT_DISPOSE_GRACE_MS = 0;

interface Job {
  id: string;
  phase: InstallPhase;
  pythonPath: string;
  command: string;
  startedAt: string;
  finishedAt?: string;
  exitCode?: number | null;
  logPath: string;
  child: InstallChild;
  /** Set between a termination request and the pip process actually exiting. */
  cancelling: boolean;
  /**
   * Escalation timer that force-kills the child after the grace period.
   *
   * It never settles the job: only an observed exit does.
   */
  terminateTimer?: ReturnType<typeof setTimeout>;
}

export interface InstallJobsOptions {
  spawner: InstallSpawner;
  logDir: string;
  /**
   * Grace period between the graceful termination signal and the force-kill
   * of a cancelled job. Defaults to 5 s, so pip can clean up before it is
   * force-killed.
   */
  killGraceMs?: number;
  /**
   * Grace period between the graceful termination signal and the force-kill
   * of a disposed job. Defaults to 0: the plugin is unloading, so the
   * force-kill follows immediately and a child can never outlive the Host.
   */
  disposeGraceMs?: number;
}

/**
 * One long-running pip install at a time per interpreter. Jobs live in Host
 * memory and write a log file, so closing the browser keeps them running and
 * a running job can be restored by polling the latest status.
 */
export class InstallJobs {
  private readonly jobs = new Map<string, Job>();
  private latest: string | undefined;
  private sequence = 0;
  private readonly killGraceMs: number;
  private readonly disposeGraceMs: number;

  constructor(private readonly options: InstallJobsOptions) {
    this.killGraceMs = options.killGraceMs ?? DEFAULT_KILL_GRACE_MS;
    this.disposeGraceMs = options.disposeGraceMs ?? DEFAULT_DISPOSE_GRACE_MS;
  }

  start(request: InstallRequest): InstallStatus {
    const pythonPath = (request.pythonPath ?? "").trim();
    // The interpreter may be written for another platform than the host, so its
    // own flavor decides whether the path is absolute.
    if (!pythonPath || !isAbsolutePath(pythonPath))
      throw new TypeError(
        "Install requires an absolute Python interpreter path",
      );
    // The same lookup the status view uses: a job that is still running for
    // this interpreter owns it, even when its termination is unconfirmed.
    const running = this.runningFor(pythonPath);
    if (running) return this.snapshot(running);
    const args = pipArgs(request);
    const id = `install-${Date.now().toString(
      36,
    )}-${(this.sequence += 1).toString(36)}`;
    mkdirSync(this.options.logDir, { recursive: true });
    const logPath = join(this.options.logDir, `${id}.log`);
    const child = this.options.spawner.start({
      exe: pythonPath,
      args,
      logPath,
    });
    const job: Job = {
      id,
      phase: "running",
      pythonPath,
      command: [pythonPath, ...args].join(" "),
      startedAt: new Date().toISOString(),
      logPath,
      child,
      cancelling: false,
    };
    this.jobs.set(id, job);
    this.latest = id;
    // Prune once the new log exists, so the steady state really is `keep`
    // files rather than one more than the documented number.
    pruneInstallLogs(
      this.options.logDir,
      INSTALL_LOG_KEEP,
      [...this.jobs.values()]
        .filter((running) => running.phase === "running")
        .map((running) => running.logPath),
    );
    child.onExit((code) => this.finish(job, code));
    return this.snapshot(job);
  }

  /**
   * Status of one job, of the newest running job for an interpreter, or of the
   * newest job overall.
   *
   * The interpreter lookup is what lets a tab restore the job it lost: two
   * interpreters may install at the same time, so the newest job overall is
   * not necessarily the one on screen.
   */
  status(jobId?: string, pythonPath?: string): InstallStatus | null {
    const trimmed = (jobId ?? "").trim();
    if (trimmed) {
      const job = this.jobs.get(trimmed);
      return job ? this.snapshot(job) : null;
    }
    const wanted = (pythonPath ?? "").trim();
    if (wanted) {
      // Only a running job is restored: a finished one belongs to the tab that
      // started it, which shows its own result for it.
      const job = this.runningFor(wanted);
      return job ? this.snapshot(job) : null;
    }
    const job = this.jobs.get(this.key(jobId));
    return job ? this.snapshot(job) : null;
  }

  /** The newest running job for one interpreter, if there is one. */
  private runningFor(pythonPath: string): Job | undefined {
    let found: Job | undefined;
    // The map keeps insertion order, so the last match is the newest job.
    for (const job of this.jobs.values()) {
      if (job.phase !== "running") continue;
      if (!sameInterpreter(job.pythonPath, pythonPath)) continue;
      found = job;
    }
    return found;
  }

  /**
   * Ask pip to stop. The job keeps the interpreter locked (phase stays
   * `running`, `cancelling` set) until the process has actually exited, so a
   * second installation cannot start while the first is still dying.
   *
   * A process that ignores the graceful signal is force-killed after the
   * grace period. The escalation never settles the job: only an observed
   * exit does, so a failed termination stays visible instead of being
   * reported as a completed cancellation.
   */
  cancel(jobId?: string): boolean {
    const job = this.jobs.get(this.key(jobId));
    if (!job || job.phase !== "running" || job.cancelling) return false;
    this.terminate(job, this.killGraceMs);
    return true;
  }

  /**
   * Stop every running installation; called when the plugin unloads.
   *
   * The jobs are not settled here: only an observed exit settles them. The
   * default grace period is zero, so the force-kill follows the graceful
   * signal immediately and a child that ignores it cannot outlive the Host.
   */
  dispose(): void {
    for (const job of this.jobs.values()) {
      if (job.phase !== "running") continue;
      this.terminate(job, this.disposeGraceMs);
    }
  }

  /**
   * The one termination path, shared by cancellation and disposal.
   *
   * It sends the graceful signal first and escalates to the force-kill after
   * `graceMs`. It never settles the job: only an observed exit does, so the
   * interpreter stays locked while the process is still alive and a failed
   * termination stays visible.
   */
  private terminate(job: Job, graceMs: number): void {
    this.clearTerminateTimer(job);
    job.cancelling = true;
    job.child.kill();
    if (graceMs <= 0) {
      // No grace period: the force-kill follows immediately, and nothing is
      // left pending that could fire later or keep the Host alive.
      job.child.kill(true);
      return;
    }
    const timer = setTimeout(() => {
      job.terminateTimer = undefined;
      job.child.kill(true);
    }, graceMs);
    // A pending escalation must never keep the Host process alive.
    timer.unref();
    job.terminateTimer = timer;
  }

  /** Drop the pending escalation timer of a job. */
  private clearTerminateTimer(job: Job): void {
    if (!job.terminateTimer) return;
    clearTimeout(job.terminateTimer);
    job.terminateTimer = undefined;
  }

  private key(jobId?: string): string {
    const trimmed = (jobId ?? "").trim();
    return trimmed || this.latest || "";
  }

  /** Settle a running job exactly once, from an observed exit. */
  private finish(job: Job, code: number | null): void {
    if (job.phase !== "running") return;
    this.clearTerminateTimer(job);
    job.phase = job.cancelling
      ? "cancelled"
      : code === 0
      ? "succeeded"
      : "failed";
    job.exitCode = code;
    job.finishedAt = new Date().toISOString();
    job.cancelling = false;
  }

  /** Serialize only JSON-safe values; optional fields are omitted when unset. */
  private snapshot(job: Job): InstallStatus {
    return {
      id: job.id,
      phase: job.phase,
      pythonPath: job.pythonPath,
      command: job.command,
      startedAt: job.startedAt,
      ...(job.finishedAt === undefined ? {} : { finishedAt: job.finishedAt }),
      ...(job.exitCode === undefined ? {} : { exitCode: job.exitCode }),
      ...(job.cancelling ? { cancelling: true } : {}),
      logPath: job.logPath,
      logTail: tailLog(job.logPath),
    };
  }
}
