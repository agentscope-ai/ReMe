/**
 * Environment contract shared by the DSH Host and the Environment tab.
 *
 * Everything here crosses the `remeEnv` RPC boundary, so it stays browser-safe:
 * this module must not import a node built-in. Host-only shapes (process I/O,
 * spawners, persisted state) live beside the code that owns them.
 */

/** A local Python interpreter discovered on the DSH Host. */
export type PythonOrigin =
  | { kind: "conda"; envName: string; root: string }
  | { kind: "python.org" }
  | { kind: "windows-store" }
  | { kind: "venv"; baseExe: string }
  | { kind: "system" }
  | { kind: "unknown" };

export interface PythonEnv {
  /** Verified interpreter executable (absolute, host-native separators). */
  exe: string;
  /** "major.minor.patch" reported by the interpreter; empty when unusable. */
  version: string;
  /** True when the interpreter answered and satisfies ReMe (Python >= 3.11). */
  ok: boolean;
  origin: PythonOrigin;
  /** Human-readable discovery channels, e.g. "conda env list; PATH". */
  source: string;
  /** reme-ai installation facts, probed for the interpreter recommendation. */
  reme?: { installed: boolean; version: string; versionOk?: boolean };
}

export interface PythonDetection {
  platform: string;
  scannedAt: string;
  envs: PythonEnv[];
  /** Preferred interpreter for a local service, or "" when none is usable. */
  suggested: string;
  /** Conda installation root, or "" when conda was not found. */
  condaRoot: string;
  /** Non-fatal probe failures worth showing for transparency. */
  errors: string[];
}

/** Installed `reme-ai` facts for one interpreter. */
export interface RemeProbe {
  installed: boolean;
  version: string;
  /** True when the version satisfies the minimum this integration targets. */
  versionOk: boolean;
  /** Non-empty when the probe itself failed (spawn error, timeout). */
  error?: string;
}

export interface RemeCheck extends RemeProbe {
  /** True when the AgentScope `[core]` extras import (agentscope) works. */
  hasCore: boolean;
  /** Absolute reme CLI path, or "" when only the python wrapper is available. */
  cliPath: string;
  /** Non-empty when the interpreter itself could not be probed. */
  error: string;
}

export type InstallPhase = "running" | "succeeded" | "failed" | "cancelled";

export interface InstallRequest {
  /** Absolute interpreter that receives the packages. */
  pythonPath: string;
  /** Upgrade an existing installation to the newest satisfying version. */
  upgrade?: boolean;
  /** Reinstall even when pip already considers the requirement satisfied. */
  forceReinstall?: boolean;
  /** Optional package index URL (mirror); empty keeps the user's pip config. */
  indexUrl?: string;
}

export interface InstallStatus {
  id: string;
  phase: InstallPhase;
  pythonPath: string;
  /** Effective command line, for display and diagnostics. */
  command: string;
  startedAt: string;
  finishedAt?: string;
  exitCode?: number | null;
  logPath: string;
  /** Tail of the install log; safe to poll. */
  logTail: string;
  /** True between a cancel request and the pip process actually exiting. */
  cancelling?: boolean;
}

export interface EnvEntry {
  key: string;
  secret: boolean;
  /** True when the key has a non-empty value on disk. */
  set: boolean;
  /** Masked tail for secrets; the plain value for non-secret keys. */
  masked: string;
  /** Plain value for non-secret keys; empty for secrets. */
  value: string;
}

export interface EnvSummary {
  workspaceDir: string;
  envPath: string;
  exists: boolean;
  entries: EnvEntry[];
}

export interface LlmTestResult {
  ok: boolean;
  status: number;
  latencyMs: number;
  error: string;
}

/** Stable reason a health probe failed, so the client can localize it. */
export type HealthErrorCode =
  | "http-status"
  | "not-reme"
  | "reme-error"
  | "unreachable";

/** Whether the configured endpoint answers, probed from the DSH Host. */
export interface ConnectionHealth {
  endpoint: string;
  ok: boolean;
  version: string;
  /** Non-empty when the endpoint answered but not as a ReMe service. */
  error: string;
  /** Stable reason; the client shows localized copy instead of `error`. */
  code?: HealthErrorCode;
  checkedAt: string;
}

export interface ServiceState {
  phase: "running" | "stopped" | "conflict";
  /** True when the running service was started by this plugin. */
  managed: boolean;
  endpoint: string;
  pid?: number;
  version?: string;
  startedAt?: string;
  logPath: string;
}

/** Stable reason for a failed start, so the client can localize it. */
export type ServiceStartErrorCode =
  | "other-endpoint-starting"
  | "conflict"
  | "other-endpoint-running"
  | "previous-not-stopped"
  | "unverified-record"
  | "no-python"
  | "not-installed"
  | "version-too-old"
  | "start-failed";

/** Stable reason for a refused/unsuccessful stop, so the client can localize it. */
export type ServiceStopReasonCode =
  | "no-record"
  | "foreign"
  | "unverified"
  | "not-exited"
  | "record-changed";

/**
 * Identity of the service record an action was refused on behalf of.
 *
 * The record is not necessarily the endpoint on screen: a launch for another
 * port can block the start of this one. The Host names the record it could not
 * act on, so the recovery action can target that record instead of guessing.
 */
export interface ServiceRecordRef {
  /** Endpoint the record was written for. */
  endpoint: string;
  /** Recorded pid, when the record has one. */
  pid?: number;
}

export interface ServiceStartResult {
  started: boolean;
  alreadyRunning: boolean;
  pid?: number;
  version?: string;
  command: string;
  logPath: string;
  startedAt: string;
  /** English diagnostic; shown only when no known `code` applies. */
  error?: string;
  code?: ServiceStartErrorCode;
  /** The record that blocked the start, when one did. */
  record?: ServiceRecordRef;
}

export interface ServiceLog {
  logPath: string;
  lines: string;
}

/**
 * Outcome of one Stop request. It crosses the `remeEnv` boundary, so it lives
 * here rather than beside the Host-only process adapter.
 */
export interface ServiceStopResult {
  stopped: boolean;
  pid?: number;
  /** Why an unmanaged or missing service was left alone. */
  reason?: string;
  /** Stable reason the client localizes; `reason` stays for diagnostics. */
  code?: ServiceStopReasonCode;
  /** The record that blocked the stop, when one did. */
  record?: ServiceRecordRef;
}

/**
 * Outcome of dropping a service record without touching the process. This is
 * the escape hatch for a live process whose identity cannot be verified: Stop
 * and Start both refuse to act on it, so the record can only be forgotten.
 */
export interface ServiceForgetResult {
  forgotten: boolean;
  pid?: number;
  reason?: string;
  code?: ServiceStopReasonCode;
  /** The record that was refused, when one was. */
  record?: ServiceRecordRef;
}
