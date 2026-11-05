import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

import { safeError } from "../../endpoint.js";
import { detectPythonEnvs } from "./detect.js";
import { envUpdates, storedState } from "../env-fields.js";
import { canHostService } from "../dependency.js";
import { isLocalEndpoint } from "../endpoint-rules.js";
import { createInstallSpawner, InstallJobs, tailLog } from "./install.js";
import { checkReme, MINIMUM_REME, remeVersionAtLeast } from "./reme.js";
import { createNodeRunner } from "./runner.js";
import {
  createServiceIO,
  forgetService,
  processAlive,
  readServiceState,
  removeServiceState,
  serviceEndpointKey,
  serviceOriginKey,
  serviceOwnership,
  serviceRecordRef,
  serviceStatus,
  startService,
  stopService,
  writeServiceState,
} from "./service.js";
import type { PersistedService, ServiceIO } from "./service.js";
import type { ServiceOwnership } from "./service.js";
import {
  envSummary,
  mergeEnvFile,
  parseEnvFile,
  resolveWorkspaceDir,
  testLlm,
  validateEnvUpdate,
  writeEnvText,
} from "./workspace.js";
import type {
  ConnectionHealth,
  EnvSummary,
  InstallRequest,
  InstallStatus,
  LlmTestResult,
  PythonDetection,
  RemeCheck,
  ServiceForgetResult,
  ServiceLog,
  ServiceRecordRef,
  ServiceStartErrorCode,
  ServiceStartResult,
  ServiceState,
  ServiceStopResult,
} from "../types.js";

const DEFAULT_ENDPOINT = "http://127.0.0.1:2333";
const CACHE_MS = 60_000;
/** How long a spawn is treated as "already starting" while it comes up. */
const START_GRACE_MS = 120_000;

/**
 * Host-side environment manager: the one place that runs local operations and
 * hands the results to the gateway.
 *
 * Everything here runs on the machine DSH runs on, so the tab and the Host can
 * never disagree about which Python, which `.env` or which service "local"
 * means. Lifetimes are explicit:
 *
 * - Python probes and reme-ai checks are bounded one-shot commands (`timeoutMs`).
 * - A pip installation is a job in this manager's registry; {@link dispose}
 *   stops the ones still running when the plugin unloads, and its log survives
 *   in `$DSH_HOME/reme-memory`.
 * - The ReMe service is detached on purpose: it must outlive DSH, so it is not
 *   handed to DSH's subprocess provider, which kills its children on Host exit.
 *   It is stopped only through the ownership-checked adapter in `service.ts`.
 */
export class ReMeEnvManager {
  private cache?: { at: number; detection: PythonDetection };
  private readonly installs: InstallJobs;
  private readonly env: NodeJS.ProcessEnv;
  private readonly serviceIO: ServiceIO;
  private readonly serviceStatePath: string;
  /** Serializes starts so two requests cannot spawn two services. */
  private startQueue: Promise<unknown> = Promise.resolve();
  private starting?: { endpoint: string; at: number };

  constructor(
    env: NodeJS.ProcessEnv = process.env,
    serviceIO: ServiceIO = createServiceIO(),
  ) {
    this.env = env;
    this.serviceIO = serviceIO;
    const home = env.DSH_HOME?.trim() || join(homedir(), ".dsh");
    this.installs = new InstallJobs({
      spawner: createInstallSpawner(env),
      logDir: join(home, "reme-memory"),
    });
    this.serviceStatePath = join(home, "reme-memory", "service.json");
  }

  async detectPython(
    options: { refresh?: boolean } = {},
  ): Promise<PythonDetection> {
    if (!options.refresh && this.cache && Date.now() - this.cache.at < CACHE_MS)
      return this.cache.detection;
    const detection = await detectPythonEnvs({
      run: createNodeRunner(this.env),
      platform: process.platform,
      env: this.env,
      homedir: homedir(),
      exists: existsSync,
    });
    this.cache = { at: Date.now(), detection };
    return detection;
  }

  async checkReme(pythonPath: string): Promise<RemeCheck> {
    return checkReme(
      {
        run: createNodeRunner(this.env),
        platform: process.platform,
        exists: existsSync,
      },
      pythonPath,
    );
  }

  install(request: InstallRequest): InstallStatus {
    return this.installs.start(request);
  }

  /**
   * Status of one job, of the newest running job for an interpreter, or of the
   * newest job overall. The interpreter lookup is what lets a tab restore the
   * job it lost: two interpreters may install at the same time, so "the latest
   * job" is not necessarily the one on screen.
   */
  installStatus(jobId?: string, pythonPath?: string): InstallStatus | null {
    return this.installs.status(jobId, pythonPath);
  }

  installCancel(jobId?: string): boolean {
    return this.installs.cancel(jobId);
  }

  /** Stop the installations this manager owns, on plugin disposal. */
  dispose(): void {
    this.installs.dispose();
  }

  /** Masked .env projection for the selected workspace. */
  readEnv(workspaceDir: string): EnvSummary {
    const dir = resolveWorkspaceDir(workspaceDir, homedir());
    return envSummary(join(dir, ".env"), dir);
  }

  /**
   * Merge edited values into the workspace .env. The whole merged file is
   * validated, so an untouched real key still satisfies the requirement and a
   * cleared one fails here rather than at the next service start.
   */
  writeEnv(workspaceDir: string, values: Record<string, string>): EnvSummary {
    const dir = resolveWorkspaceDir(workspaceDir, homedir());
    const path = join(dir, ".env");
    let existing = "";
    try {
      existing = readFileSync(path, "utf8");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    const stored = parseEnvFile(existing);
    // The update rules are applied here as well as in the tab: a client that
    // predates them must not be able to clear a stored credential by sending
    // an empty value for it.
    const merged = mergeEnvFile(
      existing,
      envUpdates(values, storedState(stored)),
    );
    validateEnvUpdate(parseEnvFile(merged));
    writeEnvText(path, merged);
    return envSummary(path, dir);
  }

  /** One-off LLM connectivity probe reading the workspace .env. */
  async testLlm(
    workspaceDir: string,
    pythonPath: string,
  ): Promise<LlmTestResult> {
    const dir = resolveWorkspaceDir(workspaceDir, homedir());
    return testLlm(
      { run: createNodeRunner(this.env) },
      { workspaceDir: dir, pythonPath },
    );
  }

  /** Whether the configured ReMe endpoint answers, probed from this Host. */
  async health(endpoint: string, authToken = ""): Promise<ConnectionHealth> {
    const endpointValue = this.endpoint(endpoint);
    const health = await this.serviceIO.fetchHealth(endpointValue, authToken);
    return {
      endpoint: endpointValue,
      ok: health.ok,
      version: health.version,
      error: health.error,
      ...(health.code === undefined ? {} : { code: health.code }),
      checkedAt: new Date().toISOString(),
    };
  }

  /** Current service phase (running / conflict / stopped) and ownership. */
  async serviceStatus(endpoint: string, authToken = ""): Promise<ServiceState> {
    const endpointValue = this.endpoint(endpoint);
    // A record left under a path prefix of the same origin describes the
    // service this endpoint talks to. It is re-keyed to the endpoint on screen
    // first, so the tab can manage (stop) it and the start path can replace
    // it instead of treating our own service as a foreign process.
    await this.adoptLeftover(endpointValue);
    const persisted = readServiceState(this.serviceStatePath);
    // A malformed record must not turn the log path into a relative one.
    let logDir = resolveWorkspaceDir("", homedir());
    const recordedDir = persisted?.workspaceDir?.trim() ?? "";
    if (recordedDir) {
      try {
        logDir = resolveWorkspaceDir(recordedDir, homedir());
      } catch {
        // Keep the default log directory.
      }
    }
    const status = await serviceStatus(this.serviceIO, {
      endpoint: endpointValue,
      statePath: this.serviceStatePath,
      authToken,
      logPath: join(logDir, "service.log"),
    });
    if (this.starting?.endpoint === endpointValue) {
      // Running and conflict both mean the port answered: the launch is no
      // longer "in flight". Stopped without a live pid means it died before it
      // ever listened, so a retry must not be blocked either. A stopped status
      // *with* a pid is a service still booting.
      if (
        status.phase === "running" ||
        status.phase === "conflict" ||
        (status.phase === "stopped" && status.pid === undefined)
      )
        this.starting = undefined;
    }
    return status;
  }

  /**
   * Spawn `reme start` detached; the caller polls serviceStatus. Starts are
   * serialized and re-check the phase inside the lock, so two requests cannot
   * spawn two services and overwrite the one recorded pid.
   */
  serviceStart(
    endpoint: string,
    pythonPath: string,
    workspaceDir: string,
    authToken = "",
  ): Promise<ServiceStartResult> {
    return this.enqueue(() =>
      this.startLocked(endpoint, pythonPath, workspaceDir, authToken),
    );
  }

  /**
   * Stop the plugin-managed service; a foreign service stays untouched. Stop
   * joins the start queue, so a request that arrives while a launch is still
   * being recorded waits for it and then stops that launch instead of
   * reporting "nothing started".
   */
  serviceStop(endpoint: string): Promise<ServiceStopResult> {
    const endpointValue = this.endpoint(endpoint);
    return this.enqueue(() => this.stopLocked(endpointValue));
  }

  /**
   * Drop a service record without touching the process. It is the escape hatch
   * for a live process whose identity cannot be verified (Stop and Start both
   * refuse to act on it). Joins the start queue so a launch that is still being
   * recorded completes first.
   *
   * `expectPid` pins the record the caller saw: a record that was replaced in
   * the meantime is refused instead of being dropped silently.
   */
  serviceForget(
    endpoint: string,
    expectPid?: number,
  ): Promise<ServiceForgetResult> {
    const endpointValue = this.endpoint(endpoint);
    return this.enqueue(async () => {
      const result = forgetService({
        statePath: this.serviceStatePath,
        endpoint: endpointValue,
        ...(expectPid === undefined ? {} : { expectPid }),
      });
      if (result.forgotten && this.starting?.endpoint === endpointValue)
        this.starting = undefined;
      return result;
    });
  }

  /** Tail of the workspace service log. */
  serviceLog(workspaceDir: string): ServiceLog {
    const dir = resolveWorkspaceDir(workspaceDir, homedir());
    const currentPath = join(dir, "service.log");
    const current = tailLog(currentPath, 40);
    if (current) return { logPath: currentPath, lines: current };
    // Rotation moves the previous launch's crash output to `service.log.1`; if
    // the current file is still empty, that backup is the only diagnostic. The
    // returned path is the file actually read, so the UI labels it correctly.
    const rotatedPath = `${currentPath}.1`;
    const rotated = tailLog(rotatedPath, 40);
    return rotated
      ? { logPath: rotatedPath, lines: rotated }
      : { logPath: currentPath, lines: "" };
  }

  /** DSH-startup best effort: start a stopped local service when configured. */
  async autoStart(options: {
    endpoint: string;
    pythonPath: string;
    workspaceDir: string;
    authToken?: string;
  }): Promise<void> {
    // Only a root-path loopback endpoint can become a local service. A
    // configured gateway prefix (or a remote URL) is probed elsewhere, and a
    // start could at best fail after stopping a same-origin service: skip it
    // instead of reporting a failure on every DSH start.
    let launchable = false;
    try {
      launchable = isLocalEndpoint(this.endpoint(options.endpoint));
    } catch {
      launchable = false;
    }
    if (!launchable) return;
    const status = await this.serviceStatus(
      options.endpoint,
      options.authToken ?? "",
    );
    if (status.phase === "running") return;
    let pythonPath = options.pythonPath.trim();
    if (!pythonPath) {
      const detection = await this.detectPython();
      pythonPath = detection.suggested;
    }
    if (!pythonPath) return;
    const result = await this.serviceStart(
      options.endpoint,
      pythonPath,
      options.workspaceDir,
      options.authToken ?? "",
    );
    if (!result.started && !result.alreadyRunning)
      throw new Error(result.error || "Could not auto-start the ReMe service");
  }

  /**
   * Serialize service operations: a start, a stop and a forget must not
   * interleave, or two of them could act on the same record at once. A failed
   * task must not wedge the queue, so it is swallowed here.
   */
  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    const run = this.startQueue.then(task);
    this.startQueue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  /**
   * A refused start. Every reason produces the same shape, so the client reads
   * one result contract instead of guessing from the message.
   */
  private refused(
    logPath: string,
    startedAt: string,
    code: ServiceStartErrorCode,
    error: string,
    extra?: { pid?: number; record?: ServiceRecordRef },
  ): ServiceStartResult {
    return {
      started: false,
      alreadyRunning: false,
      command: "",
      logPath,
      startedAt,
      error,
      code,
      ...(extra?.pid === undefined ? {} : { pid: extra.pid }),
      ...(extra?.record === undefined ? {} : { record: extra.record }),
    };
  }

  private endpoint(raw: string): string {
    const value = raw.trim() || DEFAULT_ENDPOINT;
    let url: URL;
    try {
      url = new URL(value);
    } catch {
      throw new TypeError("Invalid ReMe endpoint");
    }
    if (url.protocol !== "http:" && url.protocol !== "https:")
      throw new TypeError("ReMe endpoint must be an http(s) URL");
    if (url.username || url.password)
      throw new TypeError("ReMe endpoint must not contain credentials");
    if (url.search || url.hash)
      throw new TypeError(
        "ReMe endpoint must not contain a query string or fragment",
      );
    // The client appends job names to the endpoint, so it never keeps a
    // trailing slash; normalising here is what makes records comparable.
    return `${url.protocol}//${url.host}${url.pathname.replace(/\/+$/, "")}`;
  }

  private async stopLocked(endpointValue: string): Promise<ServiceStopResult> {
    // A leftover record under a path prefix of the same origin is this
    // service: it is re-keyed first, so a stop that arrives before any
    // status read still acts on the right record instead of refusing it as
    // foreign.
    await this.adoptLeftover(endpointValue);
    const result = await stopService(this.serviceIO, {
      statePath: this.serviceStatePath,
      endpoint: endpointValue,
    });
    if (result.stopped && this.starting?.endpoint === endpointValue)
      this.starting = undefined;
    return result;
  }

  /**
   * The recorded service and what can be proven about it.
   *
   * The record is read in one place and its ownership decided in one place,
   * because the verdict drives three decisions: whether a start may replace
   * the process, whether the record blocks it, and which record the tab may
   * forget.
   *
   * Ownership is decided against the record's *own* endpoint: the launch
   * configures only host and port, so a record filed under a path prefix
   * still describes the service the endpoint on screen talks to.
   */
  private async recordedService(): Promise<{
    record?: PersistedService;
    ownership: ServiceOwnership;
    alive: boolean;
  }> {
    const record = readServiceState(this.serviceStatePath);
    if (!record || record.pid === undefined)
      return { ownership: "foreign", alive: false };
    if (!processAlive(this.serviceIO, record.pid))
      return { record, ownership: "foreign", alive: false };
    return {
      record,
      alive: true,
      ownership: await serviceOwnership(
        this.serviceIO,
        record,
        record.endpoint,
      ),
    };
  }

  /** Whether two endpoints name the same local service (host and port). */
  private sameOrigin(recorded: string, endpoint: string): boolean {
    return serviceOriginKey(recorded) === serviceOriginKey(endpoint);
  }

  /**
   * Whether the recorded service is the one the endpoint on screen talks to:
   * the same endpoint, or the same local service under another path.
   */
  private relatedRecord(recorded: string, endpoint: string): boolean {
    return (
      serviceEndpointKey(recorded) === serviceEndpointKey(endpoint) ||
      this.sameOrigin(recorded, endpoint)
    );
  }

  /**
   * Re-key a leftover same-origin record to the endpoint on screen.
   *
   * Only the endpoint the record is filed under changes: it stays the record
   * of the same process, so the tab can stop it and the start path can replace
   * it. A failed write is not fatal — the start path can still replace the
   * record by its own endpoint.
   *
   * Only a record that is provably ours and describes the same local service
   * is re-keyed: the same origin, a live pid, a Python image and a command
   * line that carries the recorded `workspace_dir` and `service.port`. A
   * request that cannot launch anything never qualifies: a prefixed request
   * is refused by `buildStartCommand`, so stopping a healthy same-origin
   * service first would leave nothing running (autoStart would do that on
   * every DSH start).
   *
   * The record is compared and written without yielding in between: the
   * ownership probe above takes long enough for a launch to record a new pid,
   * and overwriting that record would orphan the fresh service. The compare
   * and the write are synchronous, so nothing can slip in between them.
   */
  private async adoptLeftover(endpointValue: string): Promise<void> {
    if (!isLocalEndpoint(endpointValue)) return;
    const { record, ownership } = await this.recordedService();
    if (!record || ownership !== "owned") return;
    // A record filed under the endpoint on screen is not a leftover.
    if (
      serviceEndpointKey(record.endpoint) === serviceEndpointKey(endpointValue)
    )
      return;
    if (!this.sameOrigin(record.endpoint, endpointValue)) return;
    const current = readServiceState(this.serviceStatePath);
    if (
      !current ||
      current.pid !== record.pid ||
      current.endpoint !== record.endpoint
    )
      return;
    try {
      writeServiceState(this.serviceStatePath, {
        ...current,
        endpoint: endpointValue,
      });
    } catch {
      // The record keeps its old endpoint; the start path replaces it anyway.
    }
  }

  private async startLocked(
    endpoint: string,
    pythonPath: string,
    workspaceDir: string,
    authToken: string,
  ): Promise<ServiceStartResult> {
    const endpointValue = this.endpoint(endpoint);
    const dir = resolveWorkspaceDir(workspaceDir, homedir());
    const logPath = join(dir, "service.log");
    const now = new Date().toISOString();
    const starting = this.starting;
    if (starting && Date.now() - starting.at < START_GRACE_MS) {
      if (starting.endpoint === endpointValue)
        // A genuine launch is in flight. Observing it is not an error: the
        // client keeps polling just as the first request does.
        return {
          started: false,
          alreadyRunning: true,
          command: "",
          logPath,
          startedAt: now,
        };
      // The service record is per install, not per endpoint; spawning a second
      // endpoint's service now would overwrite the first one's record and
      // orphan it.
      return this.refused(
        logPath,
        now,
        "other-endpoint-starting",
        `A service start for ${starting.endpoint} is still in progress`,
      );
    }
    this.starting = undefined;
    const current = await this.serviceStatus(endpointValue, authToken);
    if (current.phase === "running")
      return {
        started: false,
        alreadyRunning: true,
        command: "",
        logPath,
        startedAt: now,
        ...(current.version ? { version: current.version } : {}),
      };
    const { record, ownership, alive } = await this.recordedService();
    const replace =
      record !== undefined &&
      ownership === "owned" &&
      isLocalEndpoint(endpointValue) &&
      this.sameOrigin(record.endpoint, endpointValue);
    // A port conflict is replaceable only when this record proves that our
    // own service holds it. An unverified related record retains Forget.
    if (
      current.phase === "conflict" &&
      !replace &&
      !(
        record &&
        ownership === "unknown" &&
        this.relatedRecord(record.endpoint, endpointValue)
      )
    )
      return this.refused(
        logPath,
        now,
        "conflict",
        "Another process is already listening on this endpoint",
      );
    if (record && alive) {
      if (ownership === "unknown")
        return this.refused(
          logPath,
          now,
          "unverified-record",
          `The recorded service for ${record.endpoint} (pid ${record.pid}) could not be verified; leaving it untouched`,
          { pid: record.pid, record: serviceRecordRef(record) },
        );
      if (ownership === "owned" && !replace)
        return this.refused(
          logPath,
          now,
          "other-endpoint-running",
          `A service for ${record.endpoint} is still running; stop it from that endpoint first`,
        );
    }
    const path = pythonPath.trim();
    if (!path)
      return this.refused(
        logPath,
        now,
        "no-python",
        "Select a Python interpreter first",
      );
    const check = await this.checkReme(path);
    // The same rule the tab uses to offer a start decides it here, so a
    // button is never offered for an interpreter this refuses.
    if (!canHostService(check)) {
      const code = !check.installed
        ? "not-installed"
        : !check.versionOk
        ? "version-too-old"
        : "core-missing";
      const errors = {
        "not-installed": "reme-ai is not installed in the selected interpreter",
        "version-too-old": `reme-ai ${
          check.version || "(unknown version)"
        } is older than ${MINIMUM_REME}; upgrade it first`,
        "core-missing":
          "AgentScope is required to start ReMe; install reme-ai[core] in the selected interpreter",
      };
      return this.refused(logPath, now, code, errors[code]);
    }
    // All replacement cases share this path. Validate the new interpreter
    // before stopping anything, then re-verify ownership at the stop boundary.
    if (replace && record) {
      const previous = await stopService(this.serviceIO, {
        statePath: this.serviceStatePath,
        endpoint: record.endpoint,
        expectPid: record.pid,
      });
      const remaining = readServiceState(this.serviceStatePath);
      if (!previous.stopped || remaining)
        return this.refused(
          logPath,
          now,
          "previous-not-stopped",
          `A previous launch could not be stopped: ${
            previous.reason ?? "the service record changed while stopping"
          }`,
          { record: previous.record ?? serviceRecordRef(remaining) },
        );
    } else if (record) {
      // Only dead or provably foreign records reach here.
      removeServiceState(this.serviceStatePath);
    }
    // Set before the spawn so a second request arriving while the service is
    // booting observes the first launch instead of spawning another one. It is
    // cleared again when the spawn fails, so a retry is possible immediately.
    this.starting = { endpoint: endpointValue, at: Date.now() };
    try {
      const result = await startService(this.serviceIO, {
        endpoint: endpointValue,
        workspaceDir: dir,
        pythonPath: path,
        cliPath: check.cliPath,
        // 0.4.0.x has no jobs.<name>.enabled; only newer releases accept it.
        disableDreamCron: remeVersionAtLeast(check.version, "0.4.1"),
        statePath: this.serviceStatePath,
        logPath,
        env: this.env,
      });
      if (!result.started) this.starting = undefined;
      return result;
    } catch (error) {
      this.starting = undefined;
      return this.refused(
        logPath,
        now,
        "start-failed",
        `Could not start the ReMe service: ${safeError(error)}`,
      );
    }
  }
}
