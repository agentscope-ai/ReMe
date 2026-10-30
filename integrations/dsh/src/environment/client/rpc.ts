import type { StatusRpc } from "../../client/status-page.js";
import type {
  ConnectionHealth,
  EnvSummary,
  InstallStatus,
  LlmTestResult,
  PythonDetection,
  RemeCheck,
  ServiceForgetResult,
  ServiceLog,
  ServiceStartResult,
  ServiceState,
  ServiceStopResult,
} from "../types.js";

/** Methods the Host exposes on the `remeEnv` wire namespace. */
export type EnvMethod =
  | "python"
  | "checkReme"
  | "install"
  | "installStatus"
  | "installCancel"
  | "readEnv"
  | "writeEnv"
  | "testLlm"
  | "health"
  | "serviceStatus"
  | "serviceStart"
  | "serviceStop"
  | "serviceForget"
  | "serviceLog";

/**
 * Call one environment method. DSH exposes the SRC gateway with a single plain
 * `input` parameter, so every call sends `{ args: { input } }`.
 */
export async function envCall(
  rpc: StatusRpc,
  method: EnvMethod,
  input: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<unknown> {
  const result = await rpc.call(
    "/api",
    `remeEnv/${method}`,
    { args: { input } },
    signal,
  );
  if (!result.ok) throw new Error(result.error.message || result.error.code);
  return result.value;
}

function record(value: unknown, what: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error(`Invalid ${what} response`);
  return value as Record<string, unknown>;
}

function fail(what: string): never {
  throw new Error(`Invalid ${what} response`);
}

function stringField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): string {
  const value = body[key];
  if (typeof value !== "string") fail(what);
  return value;
}

function booleanField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): boolean {
  const value = body[key];
  if (typeof value !== "boolean") fail(what);
  return value;
}

function optionalStringField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  if (body[key] !== undefined && typeof body[key] !== "string") fail(what);
}

function optionalPidField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  const value = body[key];
  if (value === undefined) return;
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 1)
    fail(what);
}

/**
 * The record a refused action was about. The Host names it so the client can
 * offer a recovery for that record, which is not necessarily the endpoint the
 * request was made for.
 */
function optionalRecordField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): void {
  const value = body[key];
  if (value === undefined) return;
  const entry = record(value, what);
  stringField(entry, "endpoint", what);
  optionalPidField(entry, "pid", what);
}

function arrayField(
  body: Record<string, unknown>,
  key: string,
  what: string,
): unknown[] {
  const value = body[key];
  if (!Array.isArray(value)) fail(what);
  return value;
}

/**
 * Guard a detection payload. The renderers dereference `errors`, `origin` and
 * the per-interpreter fields, so a Host contract drift must surface as an
 * error in the section, not as a blank page.
 */
export function parseDetection(value: unknown): PythonDetection {
  const what = "environment detection";
  const body = record(value, what);
  stringField(body, "platform", what);
  stringField(body, "scannedAt", what);
  stringField(body, "suggested", what);
  stringField(body, "condaRoot", what);
  arrayField(body, "errors", what);
  for (const item of arrayField(body, "envs", what)) {
    const env = record(item, what);
    stringField(env, "exe", what);
    stringField(env, "version", what);
    booleanField(env, "ok", what);
    const origin = record(env.origin, what);
    const kind = stringField(origin, "kind", what);
    if (
      ![
        "conda",
        "python.org",
        "windows-store",
        "venv",
        "system",
        "unknown",
      ].includes(kind)
    )
      fail(what);
    if (kind === "conda") {
      stringField(origin, "envName", what);
      stringField(origin, "root", what);
    }
    if (kind === "venv") stringField(origin, "baseExe", what);
    if (env.reme !== undefined) {
      const reme = record(env.reme, what);
      booleanField(reme, "installed", what);
      stringField(reme, "version", what);
    }
  }
  return value as PythonDetection;
}

export function parseRemeCheck(value: unknown): RemeCheck {
  const what = "ReMe check";
  const body = record(value, what);
  booleanField(body, "installed", what);
  booleanField(body, "versionOk", what);
  booleanField(body, "hasCore", what);
  stringField(body, "version", what);
  stringField(body, "cliPath", what);
  stringField(body, "error", what);
  return value as unknown as RemeCheck;
}

/** `null` is the valid "no job yet" answer, so it is not an error. */
export function parseInstallStatus(value: unknown): InstallStatus | null {
  if (value === null) return null;
  const what = "install status";
  const body = record(value, what);
  stringField(body, "id", what);
  stringField(body, "pythonPath", what);
  stringField(body, "startedAt", what);
  stringField(body, "logPath", what);
  stringField(body, "logTail", what);
  if (body.cancelling !== undefined) booleanField(body, "cancelling", what);
  const phase = stringField(body, "phase", what);
  if (!["running", "succeeded", "failed", "cancelled"].includes(phase))
    fail(what);
  return value as unknown as InstallStatus;
}

export function parseEnvSummary(value: unknown): EnvSummary {
  const what = ".env summary";
  const body = record(value, what);
  stringField(body, "workspaceDir", what);
  stringField(body, "envPath", what);
  booleanField(body, "exists", what);
  for (const item of arrayField(body, "entries", what)) {
    const entry = record(item, what);
    stringField(entry, "key", what);
    booleanField(entry, "secret", what);
    booleanField(entry, "set", what);
    stringField(entry, "masked", what);
    stringField(entry, "value", what);
  }
  return value as unknown as EnvSummary;
}

export function parseLlmResult(value: unknown): LlmTestResult {
  const what = "LLM test";
  const body = record(value, what);
  booleanField(body, "ok", what);
  if (typeof body.status !== "number" || typeof body.latencyMs !== "number")
    fail(what);
  stringField(body, "error", what);
  return value as unknown as LlmTestResult;
}

export function parseHealth(value: unknown): ConnectionHealth {
  const what = "health";
  const body = record(value, what);
  booleanField(body, "ok", what);
  stringField(body, "version", what);
  stringField(body, "error", what);
  stringField(body, "endpoint", what);
  optionalStringField(body, "code", what);
  return value as unknown as ConnectionHealth;
}

export function parseServiceState(value: unknown): ServiceState {
  const what = "service status";
  const body = record(value, what);
  const phase = stringField(body, "phase", what);
  if (!["running", "stopped", "conflict"].includes(phase)) fail(what);
  booleanField(body, "managed", what);
  stringField(body, "endpoint", what);
  stringField(body, "logPath", what);
  optionalPidField(body, "pid", what);
  optionalStringField(body, "version", what);
  optionalStringField(body, "startedAt", what);
  return value as unknown as ServiceState;
}

export function parseServiceStart(value: unknown): ServiceStartResult {
  const what = "service start";
  const body = record(value, what);
  booleanField(body, "started", what);
  booleanField(body, "alreadyRunning", what);
  stringField(body, "command", what);
  stringField(body, "logPath", what);
  stringField(body, "startedAt", what);
  optionalStringField(body, "error", what);
  optionalStringField(body, "code", what);
  optionalStringField(body, "version", what);
  optionalPidField(body, "pid", what);
  optionalRecordField(body, "record", what);
  return value as unknown as ServiceStartResult;
}

export function parseServiceStop(value: unknown): ServiceStopResult {
  const what = "service stop";
  const body = record(value, what);
  booleanField(body, "stopped", what);
  optionalStringField(body, "reason", what);
  optionalStringField(body, "code", what);
  optionalPidField(body, "pid", what);
  optionalRecordField(body, "record", what);
  return value as unknown as ServiceStopResult;
}

export function parseServiceForget(value: unknown): ServiceForgetResult {
  const what = "service forget";
  const body = record(value, what);
  booleanField(body, "forgotten", what);
  optionalStringField(body, "reason", what);
  optionalStringField(body, "code", what);
  optionalPidField(body, "pid", what);
  optionalRecordField(body, "record", what);
  return value as unknown as ServiceForgetResult;
}

export function parseServiceLog(value: unknown): ServiceLog {
  const what = "service log";
  const body = record(value, what);
  stringField(body, "lines", what);
  stringField(body, "logPath", what);
  return value as unknown as ServiceLog;
}
