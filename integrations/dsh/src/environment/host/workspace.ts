import {
  chmodSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";

import { safeError } from "../../endpoint.js";
import { MANAGED_KEYS, SECRET_KEYS } from "../env-fields.js";
import { DEFAULT_LLM_BACKEND, llmBackendOf } from "../llm-backend.js";
import type { EnvSummary, LlmTestResult } from "../types.js";
import type { RunCommand } from "./runner.js";

/** Empty means "~/.reme"; "~" expands against home; relative paths are rejected. */
export function resolveWorkspaceDir(raw: string, home: string): string {
  const trimmed = (raw ?? "").trim();
  const expanded = trimmed.startsWith("~")
    ? join(home, trimmed.slice(1).replace(/^[\\/]+/, ""))
    : trimmed;
  const value = expanded || join(home, ".reme");
  if (!isAbsolute(value))
    throw new TypeError("ReMe workspace directory must be an absolute path");
  return resolve(value);
}

/** Parse KEY=VALUE lines; comments, blanks, quotes and unknown keys are tolerated. */
export function parseEnvFile(text: string): Record<string, string> {
  const entries: Record<string, string> = {};
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const equals = trimmed.indexOf("=");
    if (equals <= 0) continue;
    const key = trimmed.slice(0, equals).trim();
    let value = trimmed.slice(equals + 1).trim();
    if (
      value.length >= 2 &&
      ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'")))
    )
      value = value.slice(1, -1);
    entries[key] = value;
  }
  return entries;
}

/**
 * Rewrite only the lines whose keys appear in updates; every other line
 * (comments, unknown variables, ordering) is preserved byte-for-byte, and the
 * file keeps its own line ending.
 *
 * A key that appears more than once is rewritten in every position. Parsing
 * keeps the last occurrence, so rewriting only the first would leave an older
 * value in force: after this function every duplicate agrees.
 */
export function mergeEnvFile(
  existingText: string,
  updates: Record<string, string>,
): string {
  const pending = new Map(Object.entries(updates));
  const eol = /\r\n/.test(existingText) ? "\r\n" : "\n";
  const lines = existingText.length ? existingText.split(/\r?\n/) : [];
  const output: string[] = [];
  const written = new Set<string>();
  for (const line of lines) {
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=/.exec(line);
    const key = match?.[1] ?? "";
    if (key && pending.has(key)) {
      // Every occurrence takes the new value: parsing keeps the last one, so
      // rewriting only the first would leave an older value in force.
      output.push(`${key}=${pending.get(key) ?? ""}`);
      written.add(key);
      continue;
    }
    output.push(line);
  }
  // Normalize the trailing newline before appending new keys.
  while (output.length && output[output.length - 1] === "") output.pop();
  for (const [key, value] of pending)
    if (!written.has(key)) output.push(`${key}=${value}`);
  return `${output.join(eol)}${eol}`;
}

/** "••••" plus the last four characters; empty values stay empty. */
export function maskSecret(value: string): string {
  if (!value) return "";
  return value.length > 4 ? `••••${value.slice(-4)}` : "••••";
}

/**
 * Read-only projection for the tab: secrets are never returned in clear.
 *
 * The managed keys and their order come from the shared metadata, so the Host
 * projection and the browser form list the same keys in the same order.
 */
export function envSummary(path: string, workspaceDir: string): EnvSummary {
  let text = "";
  let exists = false;
  try {
    text = readFileSync(path, "utf8");
    exists = true;
  } catch {
    // A missing .env is the valid first-run state.
  }
  const parsed = parseEnvFile(text);
  const entries = MANAGED_KEYS.map((key) => {
    const value = parsed[key] ?? "";
    const secret = SECRET_KEYS.has(key);
    return {
      key,
      secret,
      set: value !== "",
      masked: secret ? maskSecret(value) : value,
      value: secret ? "" : value,
    };
  });
  return { workspaceDir, envPath: path, exists, entries };
}

/**
 * Reject a .env that cannot start ReMe. Version 0.4.0.x instantiates the
 * embedding component unconditionally, so an empty key aborts startup.
 */
export function validateEnvUpdate(values: Record<string, string>): void {
  if (!(values["LLM_API_KEY"] ?? "").trim())
    throw new TypeError("LLM_API_KEY is required");
  if (!(values["EMBEDDING_API_KEY"] ?? "").trim())
    throw new TypeError(
      "EMBEDDING_API_KEY must be non-empty (use EMPTY when vector search is disabled)",
    );
}

/**
 * Write a fully merged `.env`. The caller merges and validates first, so the
 * file on disk is exactly what was validated.
 *
 * The write is atomic (same-directory temp file plus rename) and keeps the
 * previous file as `.env.bak`: a crash or a full disk must not take unrelated
 * keys such as `LLM_API_KEY` with it. New files are owner-only (0600) because
 * they hold credentials.
 */
export function writeEnvText(path: string, text: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, text, { encoding: "utf8", mode: 0o600 });
  try {
    if (existsSync(path)) {
      const backup = `${path}.bak`;
      copyFileSync(path, backup);
      // The backup holds the same credentials, so it must not be more readable
      // than the file it preserves.
      try {
        chmodSync(backup, 0o600);
      } catch {
        // Windows has no POSIX mode bits; the copy is still user-owned.
      }
    }
    renameSync(temporary, path);
  } catch (error) {
    rmSync(temporary, { force: true });
    throw error;
  }
}

/**
 * How one provider expects to be called.
 *
 * The tab offers native Anthropic, Gemini and Ollama backends next to the
 * OpenAI-compatible ones, so the probe has to speak each provider's own
 * protocol: an OpenAI-style request would report a working provider-native
 * configuration as broken.
 */
export interface LlmProtocol {
  /** Path appended to the base URL; `{model}` is the model name. */
  path: string;
  /** Header that carries the API key; empty when the provider needs none. */
  authHeader: string;
  /** Value prefix for the key header, e.g. `Bearer `. */
  authPrefix: string;
  /** Headers every request carries. */
  headers: Record<string, string>;
  /** JSON body template; `{model}` is the model name. */
  body: string;
  /** Whether the provider needs an API key at all. */
  keyRequired: boolean;
  /**
   * Where the base URL comes from.
   *
   * ReMe's credentials decide this. The OpenAI-compatible providers and
   * Anthropic read `LLM_BASE_URL` from the workspace `.env`, so the probe
   * reads it too. Gemini, Ollama and xAI ignore that variable — their
   * credentials carry their own endpoint — so the probe must use that one as
   * well: a stale value left in the `.env` would otherwise send the probe to
   * a host the service never uses, and report a working configuration as
   * broken.
   */
  baseSource: "env" | "provider";
  /** The provider's own endpoint; used when `baseSource` is `provider`. */
  providerBase?: string;
}

/**
 * The OpenAI-compatible chat request, shared by the compatible providers.
 *
 * `{model}` is replaced with the JSON-quoted model name, so the placeholder
 * carries no quotes of its own.
 */
const OPENAI_CHAT_BODY =
  '{"model":{model},"messages":[{"role":"user","content":"ping"}],"max_tokens":1}';

/** The protocol ReMe itself falls back to when `.env` names no backend. */
const OPENAI_PROTOCOL: LlmProtocol = {
  path: "/chat/completions",
  authHeader: "Authorization",
  authPrefix: "Bearer ",
  headers: {},
  body: OPENAI_CHAT_BODY,
  keyRequired: true,
  baseSource: "env",
};

/** One entry per backend the tab offers; the key is the `.env` value. */
export const LLM_PROTOCOLS: Record<string, LlmProtocol> = {
  openai: OPENAI_PROTOCOL,
  dashscope: OPENAI_PROTOCOL,
  deepseek: OPENAI_PROTOCOL,
  moonshot: OPENAI_PROTOCOL,
  anthropic: {
    path: "/v1/messages",
    authHeader: "x-api-key",
    authPrefix: "",
    headers: { "anthropic-version": "2023-06-01" },
    body: '{"model":{model},"max_tokens":1,"messages":[{"role":"user","content":"ping"}]}',
    keyRequired: true,
    baseSource: "env",
  },
  gemini: {
    path: "/v1beta/models/{model}:generateContent",
    authHeader: "x-goog-api-key",
    authPrefix: "",
    headers: {},
    body: '{"contents":[{"role":"user","parts":[{"text":"ping"}]}],"generationConfig":{"maxOutputTokens":1}}',
    keyRequired: true,
    baseSource: "provider",
    providerBase: "https://generativelanguage.googleapis.com",
  },
  ollama: {
    path: "/api/chat",
    authHeader: "",
    authPrefix: "",
    headers: {},
    body: '{"model":{model},"messages":[{"role":"user","content":"ping"}],"stream":false}',
    keyRequired: false,
    baseSource: "provider",
    providerBase: "http://localhost:11434",
  },
  xai: {
    // ReMe reaches xAI through its gRPC SDK, so this OpenAI-compatible
    // endpoint is a connectivity approximation for that backend. It is still
    // the documented REST API of the same provider and the same key, and it
    // keeps the probe from reporting a working configuration as broken.
    path: "/chat/completions",
    authHeader: "Authorization",
    authPrefix: "Bearer ",
    headers: {},
    body: OPENAI_CHAT_BODY,
    keyRequired: true,
    baseSource: "provider",
    providerBase: "https://api.x.ai/v1",
  },
};

/** The backend ReMe itself falls back to when `.env` names none. */
export { DEFAULT_LLM_BACKEND };

/** The protocol a backend speaks; unknown backends use ReMe's default. */
export function llmProtocolFor(backend: string): LlmProtocol {
  return LLM_PROTOCOLS[llmBackendOf(backend)] ?? OPENAI_PROTOCOL;
}

/**
 * The probe program for one provider, run by the selected interpreter inside
 * the workspace so it reads the same `.env` the service would discover. The
 * provider's protocol is baked into the program, so the probe cannot fall back
 * to an OpenAI-style request for a provider that does not speak it.
 */
export function llmProbeSnippet(protocol: LlmProtocol): string {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...protocol.headers,
  };
  const auth = protocol.authHeader
    ? `headers[${JSON.stringify(protocol.authHeader)}] = ${JSON.stringify(
        protocol.authPrefix,
      )} + key\n`
    : "";
  // A provider whose own credentials ignore LLM_BASE_URL is probed with the
  // provider's endpoint, exactly as the service would call it. Reading the
  // variable anyway would send the probe to a host the service never uses.
  const base =
    protocol.baseSource === "provider"
      ? `base = ${JSON.stringify(protocol.providerBase ?? "")}\n`
      : 'base = env.get("LLM_BASE_URL", "").strip().rstrip("/")\n';
  const missingBase =
    protocol.baseSource === "provider"
      ? ""
      : 'if not base:\n    missing.append("LLM_BASE_URL")\n';
  return `import json, time, urllib.request, urllib.error, urllib.parse
env = {}
try:
    with open(".env", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
except OSError:
    pass
${base}key = env.get("LLM_API_KEY", "").strip()
model = env.get("LLM_MODEL_NAME", "").strip()
missing = []
${missingBase}if not model:
    missing.append("LLM_MODEL_NAME")
if ${protocol.keyRequired ? "True" : "False"} and not key:
    missing.append("LLM_API_KEY")
if missing:
    print(json.dumps({"ok": False, "status": 0, "latencyMs": 0, "error": "LLM config incomplete (" + ", ".join(missing) + ")"}))
    raise SystemExit(0)
url = base + ${JSON.stringify(
    protocol.path,
  )}.replace("{model}", urllib.parse.quote(model, safe=""))
body = ${JSON.stringify(
    protocol.body,
  )}.replace("{model}", json.dumps(model)).encode()
headers = ${JSON.stringify(headers)}
${auth}request = urllib.request.Request(url, data=body, headers=headers, method="POST")
started = time.time()
try:
    with urllib.request.urlopen(request, timeout=20) as response:
        response.read()
        print(json.dumps({"ok": 200 <= response.status < 300, "status": response.status, "latencyMs": int((time.time() - started) * 1000), "error": ""}))
except urllib.error.HTTPError as error:
    print(json.dumps({"ok": False, "status": error.code, "latencyMs": int((time.time() - started) * 1000), "error": "HTTP " + str(error.code)}))
except Exception as error:
    print(json.dumps({"ok": False, "status": 0, "latencyMs": int((time.time() - started) * 1000), "error": str(error)[:300]}))
`;
}

/**
 * The LLM backend the workspace `.env` selects. An unreadable or incomplete
 * file names none, which leaves the probe on ReMe's default backend.
 */
export function readLlmBackend(envPath: string): string {
  let text = "";
  try {
    text = readFileSync(envPath, "utf8");
  } catch {
    return "";
  }
  return (parseEnvFile(text)["LLM_BACKEND"] ?? "").trim();
}

/** Parse the JSON line the LLM test snippet prints. */
export function parseLlmTest(text: string): LlmTestResult {
  for (const line of text.split(/\r?\n/).reverse()) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const value = JSON.parse(trimmed) as Record<string, unknown>;
      return {
        ok: value.ok === true,
        status: typeof value.status === "number" ? value.status : 0,
        latencyMs: typeof value.latencyMs === "number" ? value.latencyMs : 0,
        error: typeof value.error === "string" ? value.error : "",
      };
    } catch {
      // Try an earlier line.
    }
  }
  return {
    ok: false,
    status: 0,
    latencyMs: 0,
    error: "LLM test produced no result",
  };
}

/** Run the LLM probe with the selected interpreter inside the workspace. */
export async function testLlm(
  io: { run: RunCommand },
  options: { workspaceDir: string; pythonPath: string },
): Promise<LlmTestResult> {
  // The backend in the workspace `.env` decides the protocol; the probe must
  // not fall back to an OpenAI-style request for a provider that speaks its
  // own, or it would report a working configuration as broken.
  const protocol = llmProtocolFor(
    readLlmBackend(join(options.workspaceDir, ".env")),
  );
  try {
    const result = await io.run(
      options.pythonPath,
      ["-c", llmProbeSnippet(protocol)],
      {
        cwd: options.workspaceDir,
        timeoutMs: 30_000,
      },
    );
    if (result.error)
      return {
        ok: false,
        status: 0,
        latencyMs: 0,
        error: safeError(result.error),
      };
    const parsed = parseLlmTest(result.stdout);
    if (!parsed.ok && parsed.error) {
      const hint =
        parsed.status === 401 || parsed.status === 403
          ? "check LLM_API_KEY"
          : parsed.status === 404
          ? "check LLM_MODEL_NAME / LLM_BASE_URL"
          : /refused|timed out|getaddrinfo|Name or service/i.test(parsed.error)
          ? "check LLM_BASE_URL and network"
          : "";
      if (hint) parsed.error = `${parsed.error} (${hint})`;
    }
    return parsed;
  } catch (error) {
    return { ok: false, status: 0, latencyMs: 0, error: safeError(error) };
  }
}
