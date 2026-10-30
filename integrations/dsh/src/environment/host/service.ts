import { execFile, spawn } from "node:child_process";
import {
  closeSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { createConnection } from "node:net";
import { basename, dirname } from "node:path";

import { safeError } from "../../endpoint.js";
import { ReMeClient } from "../../reme/client.js";
import type { ReMeHealthResult } from "../../reme/types.js";
import { serviceEndpointKey } from "../endpoint-key.js";
import { parseLocalEndpoint } from "../endpoint-rules.js";
import type {
  HealthErrorCode,
  ServiceForgetResult,
  ServiceRecordRef,
  ServiceStartResult,
  ServiceState,
  ServiceStopResult,
} from "../types.js";
import { openAppendLog } from "./log.js";
import { isWindowsPath } from "./paths.js";

export { serviceEndpointKey } from "../endpoint-key.js";
export { parseLocalEndpoint } from "../endpoint-rules.js";

/**
 * The local ReMe service is a *detached* process: it outlives DSH, which is the
 * point of starting it from here, so it is deliberately not owned by DSH's
 * subprocess provider (that provider terminates its children on Host exit).
 * This module is the small service adapter that owns that lifetime instead:
 * it spawns the process, records the pid in a state file, verifies ownership
 * before reporting or stopping it, and tree-kills it on request.
 */

export interface PersistedService {
  pid?: number;
  endpoint: string;
  workspaceDir: string;
  pythonPath: string;
  command: string;
  startedAt: string;
}

export interface ServiceHealth {
  ok: boolean;
  version: string;
  /** Non-empty when the endpoint answered but not as a ReMe service. */
  error: string;
  /** Stable reason the client localizes instead of showing `error`. */
  code?: HealthErrorCode;
}

export interface ServiceIO {
  fetchHealth(endpoint: string, authToken: string): Promise<ServiceHealth>;
  portOpen(host: string, port: number): Promise<boolean>;
  spawnService(options: {
    exe: string;
    args: string[];
    cwd: string;
    logPath: string;
    env: NodeJS.ProcessEnv;
  }): Promise<{ pid: number | undefined }>;
  /**
   * Ask the process, and its process group on POSIX, to stop. `force`
   * escalates to `SIGKILL` (always `/F` on Windows).
   */
  killTree(pid: number, force?: boolean): void;
  probePid(pid: number): void;
  /** OS image name of a pid, or "" when it cannot be read. */
  processImage(pid: number): Promise<string>;
  /** Full command line of a pid, or "" when it cannot be read. */
  processCommandLine?(pid: number): Promise<string>;
  exists(path: string): boolean;
  platform: string;
}

/** A pid we are willing to signal: > 1, so kill(0)/kill(-1) is impossible. */
export function isValidPid(pid: unknown): pid is number {
  return typeof pid === "number" && Number.isInteger(pid) && pid > 1;
}

/**
 * Compose the `reme start` invocation. The working directory must be the
 * workspace so ReMe's `.env` discovery (cwd plus five parents) finds the file
 * the Environment tab writes.
 */
export function buildStartCommand(options: {
  cliPath: string;
  pythonPath: string;
  workspaceDir: string;
  endpoint: string;
  disableDreamCron: boolean;
  exists(path: string): boolean;
}): { exe: string; args: string[]; display: string } {
  const { port } = parseLocalEndpoint(options.endpoint);
  const args = [
    "start",
    `workspace_dir=${options.workspaceDir}`,
    "service.host=127.0.0.1",
    `service.port=${port}`,
    // Keeps the rich logo from crashing GBK consoles in the background.
    "enable_logo=false",
    ...(options.disableDreamCron ? ["jobs.dream_cron.enabled=false"] : []),
  ];
  if (options.cliPath && options.exists(options.cliPath))
    return {
      exe: options.cliPath,
      args,
      display: `${options.cliPath} ${args.join(" ")}`,
    };
  // The console script is missing (e.g. a bare venv without Scripts entry):
  // run the same CLI entry through the interpreter itself.
  const wrapper = `import sys;sys.argv=${JSON.stringify([
    "reme",
    ...args,
  ])};from reme.reme import main;sys.exit(main())`;
  return {
    exe: options.pythonPath,
    args: ["-c", wrapper],
    display: `${options.pythonPath} -c <reme wrapper>`,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Read a record defensively: a malformed file is "no record", not a crash. */
export function readServiceState(path: string): PersistedService | undefined {
  let value: unknown;
  try {
    value = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return undefined;
  }
  if (!isRecord(value)) return undefined;
  const endpoint = typeof value.endpoint === "string" ? value.endpoint : "";
  if (!endpoint.trim()) return undefined;
  return {
    ...(isValidPid(value.pid) ? { pid: value.pid } : {}),
    endpoint,
    workspaceDir:
      typeof value.workspaceDir === "string" ? value.workspaceDir : "",
    pythonPath: typeof value.pythonPath === "string" ? value.pythonPath : "",
    command: typeof value.command === "string" ? value.command : "",
    startedAt: typeof value.startedAt === "string" ? value.startedAt : "",
  };
}

export function writeServiceState(path: string, state: PersistedService): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(state, null, 2), {
    encoding: "utf8",
    mode: 0o600,
  });
  try {
    renameSync(temporary, path);
  } catch (error) {
    rmSync(temporary, { force: true });
    throw error;
  }
}

/** Drop the service record without touching the recorded process. */
export function removeServiceState(path: string): void {
  try {
    rmSync(path, { force: true });
  } catch {
    // A stale state file is retried by the next status read.
  }
}

/**
 * Identity of a persisted record, for a result that names the record it could
 * not act on. The recovery action targets this record, which is not
 * necessarily the endpoint the request was made for.
 */
export function serviceRecordRef(
  persisted: PersistedService | undefined,
): ServiceRecordRef | undefined {
  if (!persisted) return undefined;
  return {
    endpoint: persisted.endpoint,
    ...(persisted.pid === undefined ? {} : { pid: persisted.pid }),
  };
}

const clearServiceState = removeServiceState;

function alive(io: ServiceIO, pid?: number): boolean {
  if (!isValidPid(pid)) return false;
  try {
    io.probePid(pid);
    return true;
  } catch (error) {
    // Only ESRCH proves the process is gone. EPERM means it exists but this
    // user may not signal it (for example a service DSH started elevated);
    // treating that as dead would drop the record and orphan the process.
    return (error as NodeJS.ErrnoException | undefined)?.code === "EPERM";
  }
}

/** Whether a pid is alive, rejecting pids this plugin never records. */
export function processAlive(io: ServiceIO, pid?: number): boolean {
  return alive(io, pid);
}

/**
 * Whether an OS image name is a process that could be running the ReMe
 * service: an exact Python-family name or `reme`, never a prefix match (which
 * would accept e.g. `pycharm64.exe`).
 */
export function isServiceImage(image: string): boolean {
  const name = basename(image.trim().replaceAll("\\", "/")).toLowerCase();
  return /^(python|pythonw|reme)([0-9.]*)?(\.exe)?$/.test(name);
}

/**
 * Ownership of the recorded process, as far as this plugin can prove it.
 *
 * - `owned`: endpoint, pid, image and command line all match the launch this
 *   plugin recorded.
 * - `foreign`: at least one piece of evidence contradicts the record.
 * - `unknown`: the process exists but its image or command line could not be
 *   read (localized `tasklist`, `ps` failure, missing permissions).
 *
 * Stop is destructive, so it only accepts `owned`; starting can discard an
 * `unknown` record (without killing it) instead of being blocked forever.
 */
export type ServiceOwnership = "owned" | "foreign" | "unknown";

/**
 * Compare paths and ports with separators normalized. Windows command lines
 * escape quotes as `\"`, so that sequence is folded back before matching; case
 * is only folded for Windows paths, because POSIX paths are case-sensitive.
 */
function normalizeCommandText(value: string, foldCase: boolean): string {
  const text = value
    .replaceAll('\\"', '"')
    .replaceAll("\\\\", "\\")
    .replaceAll("/", "\\");
  return foldCase ? text.toLowerCase() : text;
}

/**
 * The recorded launch always passes `workspace_dir=<dir>` and
 * `service.port=<port>` as argv entries (also inside the Python wrapper's
 * `-c` script), so their presence is strong evidence that this pid is the
 * process the record describes — not merely another Python that reused the pid.
 */
function commandMatches(command: string, persisted: PersistedService): boolean {
  let port: number;
  try {
    port = parseLocalEndpoint(persisted.endpoint).port;
  } catch {
    return false;
  }
  const workspace = persisted.workspaceDir.trim();
  if (!workspace) return false;
  const foldCase = isWindowsPath(workspace);
  const text = normalizeCommandText(command, foldCase);
  const marker = `workspace_dir=${normalizeCommandText(workspace, foldCase)}`;
  const index = text.indexOf(marker);
  if (index < 0) return false;
  // Require an argument boundary. A path separator is *not* one: the record
  // `C:\a` must not match a launch for `C:\a\b` (POSIX `/repo` vs `/repo/sub`).
  // The `\"` form Windows shows for the wrapper is folded by
  // `normalizeCommandText`, so the quote itself is the boundary there.
  const next = text[index + marker.length] ?? "";
  if (next && !/[\s"',\]]/.test(next)) return false;
  return new RegExp(`service\\.port=${port}(?!\\d)`).test(text);
}

export async function serviceOwnership(
  io: ServiceIO,
  persisted: PersistedService | undefined,
  endpoint: string,
): Promise<ServiceOwnership> {
  if (!persisted || !isValidPid(persisted.pid)) return "foreign";
  if (serviceEndpointKey(persisted.endpoint) !== serviceEndpointKey(endpoint))
    return "foreign";
  if (!alive(io, persisted.pid)) return "foreign";
  let image = "";
  try {
    image = await io.processImage(persisted.pid);
  } catch {
    image = "";
  }
  if (!image.trim()) return "unknown";
  if (!isServiceImage(image)) return "foreign";
  let command = "";
  try {
    command = (await io.processCommandLine?.(persisted.pid)) ?? "";
  } catch {
    command = "";
  }
  if (!command.trim()) return "unknown";
  return commandMatches(command, persisted) ? "owned" : "foreign";
}

/**
 * Whether the recorded process is provably the service this plugin started
 * for the endpoint on screen.
 *
 * This guards a destructive operation, so it fails *closed*: unreadable
 * evidence is not proof of ownership.
 */
export async function ownsService(
  io: ServiceIO,
  persisted: PersistedService | undefined,
  endpoint: string,
): Promise<boolean> {
  return (await serviceOwnership(io, persisted, endpoint)) === "owned";
}

/** Running / conflict / stopped, with ownership of the running process. */
export async function serviceStatus(
  io: ServiceIO,
  options: {
    endpoint: string;
    statePath: string;
    logPath: string;
    authToken?: string;
  },
): Promise<ServiceState> {
  const health = await io.fetchHealth(
    options.endpoint,
    options.authToken ?? "",
  );
  const persisted = readServiceState(options.statePath);
  const recordForEndpoint =
    persisted !== undefined &&
    serviceEndpointKey(persisted.endpoint) ===
      serviceEndpointKey(options.endpoint);
  const recordAlive = alive(io, persisted?.pid);
  // A record whose process is gone must not be reported, or a later start
  // would keep seeing a stale pid. Remove it while nobody can be relying on it.
  if (persisted && !recordAlive) clearServiceState(options.statePath);
  // Ownership is expensive (image plus command line). During boot the service
  // is polled every second while `stopped`, where it is not needed, so it is
  // computed lazily only for the phases that report it.
  const ownership = async (): Promise<ServiceOwnership> =>
    recordForEndpoint && recordAlive
      ? serviceOwnership(io, persisted, options.endpoint)
      : "foreign";
  if (health.ok) {
    const managed = (await ownership()) === "owned";
    return {
      phase: "running",
      managed,
      endpoint: options.endpoint,
      ...(managed && persisted?.pid !== undefined
        ? { pid: persisted.pid }
        : {}),
      ...(health.version ? { version: health.version } : {}),
      ...(managed && persisted ? { startedAt: persisted.startedAt } : {}),
      logPath: options.logPath,
    };
  }
  let portOpen = false;
  try {
    const { host, port } = parseLocalEndpoint(options.endpoint);
    portOpen = await io.portOpen(host, port);
  } catch {
    // A remote or https endpoint is health-checked only; it has no local port.
  }
  if (portOpen) {
    // Our own service can occupy the port without being healthy (still
    // booting, or a rejected token); identify it so the tab can stop it.
    const managed = (await ownership()) === "owned";
    return {
      phase: "conflict",
      managed,
      endpoint: options.endpoint,
      ...(managed && persisted?.pid !== undefined
        ? { pid: persisted.pid }
        : {}),
      logPath: options.logPath,
    };
  }
  return {
    phase: "stopped",
    managed: false,
    endpoint: options.endpoint,
    // A live record for this endpoint means a launch that never became
    // healthy; the manager stops or discards it before spawning a replacement.
    ...(recordForEndpoint && recordAlive && persisted?.pid !== undefined
      ? { pid: persisted.pid }
      : {}),
    logPath: options.logPath,
  };
}

/**
 * Spawn a detached service; the caller polls status until it is healthy. The
 * launch is recorded only once the OS acknowledged the spawn, so a missing
 * interpreter leaves no state file behind for a later tab to adopt.
 */
export async function startService(
  io: ServiceIO,
  options: {
    endpoint: string;
    workspaceDir: string;
    pythonPath: string;
    cliPath: string;
    disableDreamCron: boolean;
    statePath: string;
    logPath: string;
    env: NodeJS.ProcessEnv;
  },
): Promise<ServiceStartResult> {
  const startedAt = new Date().toISOString();
  let command: { exe: string; args: string[]; display: string };
  try {
    mkdirSync(options.workspaceDir, { recursive: true });
    command = buildStartCommand({
      cliPath: options.cliPath,
      pythonPath: options.pythonPath,
      workspaceDir: options.workspaceDir,
      endpoint: options.endpoint,
      disableDreamCron: options.disableDreamCron,
      exists: io.exists,
    });
  } catch (error) {
    return {
      started: false,
      alreadyRunning: false,
      command: "",
      logPath: options.logPath,
      startedAt,
      error: `Could not start the ReMe service: ${safeError(error)}`,
      code: "start-failed",
    };
  }
  let child: { pid: number | undefined };
  try {
    child = await io.spawnService({
      exe: command.exe,
      args: command.args,
      cwd: options.workspaceDir,
      logPath: options.logPath,
      env: {
        ...options.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
        NO_PROXY: "localhost,127.0.0.1,::1",
      },
    });
  } catch (error) {
    return {
      started: false,
      alreadyRunning: false,
      command: command.display,
      logPath: options.logPath,
      startedAt,
      error: `Could not start the ReMe service: ${safeError(error)}`,
      code: "start-failed",
    };
  }
  try {
    writeServiceState(options.statePath, {
      pid: child.pid,
      endpoint: options.endpoint,
      workspaceDir: options.workspaceDir,
      pythonPath: options.pythonPath,
      command: command.display,
      startedAt,
    });
  } catch (error) {
    // The process exists but could not be recorded; leaving it running would
    // make it unmanageable, so stop it before reporting the failure.
    try {
      if (child.pid !== undefined) io.killTree(child.pid, true);
    } catch {
      // Best effort: the error below already explains the launch failed.
    }
    return {
      started: false,
      alreadyRunning: false,
      command: command.display,
      logPath: options.logPath,
      startedAt,
      error: `Could not record the ReMe service: ${safeError(error)}`,
      code: "start-failed",
    };
  }
  return {
    started: true,
    alreadyRunning: false,
    ...(child.pid === undefined ? {} : { pid: child.pid }),
    command: command.display,
    logPath: options.logPath,
    startedAt,
  };
}

const STOP_TIMEOUT_MS = 5_000;
const STOP_FORCE_TIMEOUT_MS = 3_000;
const STOP_POLL_MS = 100;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

/** Poll until the pid disappears; true when it exited within `timeoutMs`. */
async function waitForExit(
  io: ServiceIO,
  pid: number,
  timeoutMs: number,
  pollMs: number,
): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (!alive(io, pid)) return true;
    const remaining = deadline - Date.now();
    if (remaining <= 0) return false;
    await delay(Math.max(1, Math.min(pollMs, remaining)));
  }
}

/**
 * Stop the service this plugin started for the endpoint, and only that one: a
 * service on another port, or a foreign process, is left running.
 *
 * The record is removed only after the process is confirmed gone. Otherwise a
 * SIGTERM the service ignored would delete the pid, leaving a running process
 * that the tab can never stop again.
 */
export async function stopService(
  io: ServiceIO,
  options: {
    statePath: string;
    endpoint: string;
    /** Grace period before escalating to SIGKILL (tests shorten it). */
    timeoutMs?: number;
    forceTimeoutMs?: number;
    pollMs?: number;
    /** Precomputed ownership from the caller; avoids a second process probe. */
    ownership?: ServiceOwnership;
  },
): Promise<ServiceStopResult> {
  const persisted = readServiceState(options.statePath);
  if (!persisted || !isValidPid(persisted.pid)) {
    if (persisted) clearServiceState(options.statePath);
    return {
      stopped: false,
      reason: "No service started by this plugin",
      code: "no-record",
    };
  }
  const ownership =
    options.ownership ??
    (await serviceOwnership(io, persisted, options.endpoint));
  if (ownership === "unknown")
    return {
      stopped: false,
      reason:
        "The recorded process identity could not be verified; refusing to stop it",
      code: "unverified",
      record: serviceRecordRef(persisted),
    };
  if (ownership !== "owned")
    return {
      stopped: false,
      reason: "The recorded service does not belong to this endpoint",
      code: "foreign",
    };
  const pid = persisted.pid;
  const pollMs = options.pollMs ?? STOP_POLL_MS;
  try {
    io.killTree(pid);
  } catch {
    // Already exited.
  }
  if (
    await waitForExit(io, pid, options.timeoutMs ?? STOP_TIMEOUT_MS, pollMs)
  ) {
    clearServiceState(options.statePath);
    return { stopped: true, pid };
  }
  // The service ignored (or slowly handled) SIGTERM: escalate, then only drop
  // the record once the process is really gone.
  try {
    io.killTree(pid, true);
  } catch {
    // Already exited.
  }
  if (
    await waitForExit(
      io,
      pid,
      options.forceTimeoutMs ?? STOP_FORCE_TIMEOUT_MS,
      pollMs,
    )
  ) {
    clearServiceState(options.statePath);
    return { stopped: true, pid };
  }
  return {
    stopped: false,
    pid,
    reason: "The service did not exit after the force-kill",
    code: "not-exited",
    record: serviceRecordRef(persisted),
  };
}

/**
 * Drop the service record without signalling any process.
 *
 * Stop and Start both fail closed when ownership cannot be proven. With a live
 * but unverifiable process (for example one started by an elevated DSH) that
 * leaves the tab with no working action at all, so forgetting the record is the
 * explicit way out: the plugin stops managing the process and the user confirms
 * what to do with it themselves.
 *
 * `expectPid` pins the record the caller saw. Without it a record that was
 * replaced in the meantime would be dropped silently; with it the request is
 * refused instead, because the process it was meant to describe is a different
 * one now.
 */
export function forgetService(options: {
  statePath: string;
  endpoint: string;
  /** Pid the caller saw; the record must still describe it. */
  expectPid?: number;
}): ServiceForgetResult {
  const persisted = readServiceState(options.statePath);
  if (!persisted || !isValidPid(persisted.pid)) {
    if (persisted) clearServiceState(options.statePath);
    return {
      forgotten: false,
      reason: "No service started by this plugin",
      code: "no-record",
    };
  }
  if (
    serviceEndpointKey(persisted.endpoint) !==
    serviceEndpointKey(options.endpoint)
  )
    return {
      forgotten: false,
      reason: "The recorded service does not belong to this endpoint",
      code: "foreign",
    };
  if (options.expectPid !== undefined && persisted.pid !== options.expectPid)
    return {
      forgotten: false,
      pid: persisted.pid,
      reason: "The recorded service changed since it was inspected",
      code: "record-changed",
      record: serviceRecordRef(persisted),
    };
  clearServiceState(options.statePath);
  return { forgotten: true, pid: persisted.pid };
}

/** How long the Host waits for a health answer before giving up. */
const HEALTH_TIMEOUT_MS = 3_000;

/**
 * The Host's health verdict, taken from the shared ReMe client.
 *
 * The client already knows how to reach a ReMe service, how to authenticate
 * against it and how to tell one from a generic HTTP server, so the Host
 * reuses it instead of repeating that request logic. The stable codes stay:
 * the tab localizes them instead of showing an English sentence.
 */
function healthFromResult(result: ReMeHealthResult): ServiceHealth {
  const status = result.status ?? 0;
  if (status === 0)
    return {
      ok: false,
      version: "",
      error: result.error || "The endpoint could not be reached",
      code: "unreachable",
    };
  if (!result.ok) {
    // The client reports an HTTP failure and a ReMe-reported error the same
    // way; only the status tells them apart.
    const code = status >= 200 && status < 300 ? "reme-error" : "http-status";
    return {
      ok: false,
      version: "",
      error:
        code === "http-status" ? `HTTP ${status}` : "ReMe reported an error",
      code,
    };
  }
  // A 200 response is not a healthy ReMe: the health report has to parse.
  if (!result.health)
    return {
      ok: false,
      version: "",
      error: "The endpoint did not answer with a ReMe health report",
      code: "not-reme",
    };
  return { ok: true, version: result.health.version, error: "" };
}

/** Node-backed IO: fetch health, probe the port, spawn detached, tree-kill. */
export function createServiceIO(): ServiceIO {
  return {
    async fetchHealth(endpoint, authToken) {
      const client = new ReMeClient({
        endpoint,
        requestTimeoutMs: HEALTH_TIMEOUT_MS,
        backgroundTimeoutMs: HEALTH_TIMEOUT_MS,
        ...(authToken ? { authToken } : {}),
      });
      return healthFromResult(await client.healthCheck());
    },
    portOpen(host, port) {
      return new Promise((resolve) => {
        const socket = createConnection({ host, port });
        const finish = (value: boolean): void => {
          socket.destroy();
          resolve(value);
        };
        socket.setTimeout(800);
        socket.once("connect", () => finish(true));
        socket.once("timeout", () => finish(false));
        socket.once("error", () => finish(false));
      });
    },
    async spawnService({ exe, args, cwd, logPath, env }) {
      const fd = openAppendLog(logPath);
      let child: ReturnType<typeof spawn> | undefined;
      try {
        child = spawn(exe, args, {
          shell: false,
          windowsHide: true,
          detached: true,
          cwd,
          env,
          stdio: ["ignore", fd, fd],
        });
        const started = child;
        await new Promise<void>((resolve, reject) => {
          // A spawn acknowledgement that never arrives must not block the
          // start queue (and therefore a queued Stop) forever.
          const timer = setTimeout(() => {
            try {
              started.kill();
            } catch {
              // Already gone.
            }
            reject(new Error("spawn did not acknowledge within 10 s"));
          }, 10_000);
          timer.unref();
          started.once("spawn", () => {
            clearTimeout(timer);
            resolve();
          });
          started.once("error", (error: Error) => {
            clearTimeout(timer);
            reject(error);
          });
        });
      } finally {
        // The child owns its inherited handle; the parent copy is no longer
        // needed. The listener and `unref()` must run on the timeout path too:
        // otherwise a late acknowledgement keeps the child referenced and its
        // `error` event unhandled, and the Host cannot exit.
        closeSync(fd);
        if (child !== undefined) {
          child.on("error", () => undefined);
          child.unref();
        }
      }
      return { pid: child?.pid };
    },
    killTree(pid, force = false) {
      if (!isValidPid(pid)) return;
      if (process.platform === "win32") {
        const child = spawn("taskkill", ["/PID", String(pid), "/T", "/F"], {
          windowsHide: true,
          stdio: "ignore",
        });
        // A spawn failure is an async event; without a listener it would exit
        // the Host.
        child.on("error", () => undefined);
        child.unref();
        return;
      }
      // The service was spawned detached, so it leads its own process group;
      // signalling the group stops pip's children too.
      const signal = force ? "SIGKILL" : "SIGTERM";
      try {
        process.kill(-pid, signal);
        return;
      } catch {
        // Not a group leader (e.g. a record from an older launch).
      }
      try {
        process.kill(pid, signal);
      } catch {
        // Already exited.
      }
    },
    probePid(pid) {
      if (!isValidPid(pid))
        throw new RangeError("Refusing to probe an invalid pid");
      process.kill(pid, 0);
    },
    processImage(pid) {
      return new Promise((resolve) => {
        if (!isValidPid(pid)) {
          resolve("");
          return;
        }
        if (process.platform === "win32") {
          execFile(
            "tasklist",
            ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"],
            { windowsHide: true, timeout: 3000 },
            (error, stdout) => {
              const match = /^"([^"]+)"/.exec(String(stdout ?? "").trim());
              resolve(error ? "" : match?.[1] ?? "");
            },
          );
          return;
        }
        execFile(
          "ps",
          ["-p", String(pid), "-o", "comm="],
          { timeout: 3000 },
          (error, stdout) => resolve(error ? "" : String(stdout ?? "").trim()),
        );
      });
    },
    processCommandLine(pid) {
      return new Promise((resolve) => {
        if (!isValidPid(pid)) {
          resolve("");
          return;
        }
        if (process.platform === "win32") {
          // wmic is absent from recent Windows builds, so ask CIM. The encoding
          // prefix keeps non-ASCII workspace paths intact through the pipe.
          execFile(
            "powershell.exe",
            [
              "-NoProfile",
              "-NonInteractive",
              "-Command",
              `[Console]::OutputEncoding=[Text.Encoding]::UTF8; (Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`,
            ],
            { windowsHide: true, timeout: 5000 },
            (error, stdout) =>
              resolve(error ? "" : String(stdout ?? "").trim()),
          );
          return;
        }
        execFile(
          "ps",
          ["-p", String(pid), "-o", "args="],
          { timeout: 3000 },
          (error, stdout) => resolve(error ? "" : String(stdout ?? "").trim()),
        );
      });
    },
    exists: existsSync,
    platform: process.platform,
  };
}
