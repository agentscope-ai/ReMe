import { win32 } from "node:path";

import type { RemeCheck, RemeProbe } from "../types.js";
import { isAbsolutePath, pathOf } from "./paths.js";
import type { RunCommand } from "./runner.js";

/** Minimum reme-ai version this DSH integration is tested against. */
export const MINIMUM_REME = "0.4.0";

const REME_VERSION_SNIPPET =
  "import importlib.metadata as m;print(m.version('reme-ai'))";

export interface RemeIO {
  run: RunCommand;
  platform: string;
  exists(path: string): boolean;
}

/** Numeric dotted-version comparison; returns -1, 0 or 1. */
export function compareVersions(a: string, b: string): number {
  const left = a.split(".").map((part) => Number.parseInt(part, 10) || 0);
  const right = b.split(".").map((part) => Number.parseInt(part, 10) || 0);
  const length = Math.max(left.length, right.length);
  for (let index = 0; index < length; index += 1) {
    const diff = (left[index] ?? 0) - (right[index] ?? 0);
    if (diff !== 0) return Math.sign(diff);
  }
  return 0;
}

/** Strict numeric version gate; empty or prefixed versions never pass. */
export function remeVersionAtLeast(
  version: string,
  minimum: string = MINIMUM_REME,
): boolean {
  const normalized = version.trim();
  if (!/^\d+(?:\.\d+)+$/.test(normalized)) return false;
  return compareVersions(normalized, minimum) >= 0;
}

/** Read a bare version line out of pip/import noise. */
export function parseRemeVersion(text: string): string {
  const match = /^\s*(\d+(?:\.\d+)+)\s*$/m.exec(text);
  return match?.[1] ?? "";
}

/** Ask one interpreter for its installed reme-ai metadata version. */
export async function probeReme(
  io: RemeIO,
  pythonPath: string,
): Promise<RemeProbe> {
  try {
    const result = await io.run(pythonPath, ["-c", REME_VERSION_SNIPPET], {
      timeoutMs: 10_000,
    });
    if (result.error || result.code !== 0)
      return {
        installed: false,
        version: "",
        versionOk: false,
        ...(result.error ? { error: result.error } : {}),
      };
    const version = parseRemeVersion(result.stdout);
    return {
      installed: version !== "",
      version,
      versionOk: remeVersionAtLeast(version),
    };
  } catch (error) {
    return {
      installed: false,
      version: "",
      versionOk: false,
      error: String(error),
    };
  }
}

/** Locate the console script installed beside the interpreter. */
export function remeCliPath(io: RemeIO, pythonPath: string): string {
  // The interpreter path may be written for another platform than the host, so
  // its flavor supplies the separators and joins.
  const path = io.platform === "win32" ? win32 : pathOf(pythonPath);
  const directory = path.dirname(pythonPath);
  // venv interpreters live inside the scripts/bin directory; conda ones do not.
  const scriptsDir =
    io.platform === "win32"
      ? /[\\/]scripts$/i.test(directory)
        ? directory
        : path.join(directory, "Scripts")
      : directory;
  const candidate = path.join(
    scriptsDir,
    io.platform === "win32" ? "reme.exe" : "reme",
  );
  return io.exists(candidate) ? candidate : "";
}

/**
 * Full dependency check for one interpreter: reme-ai presence/version, the
 * `[core]` AgentScope import, and the CLI entry used by service management.
 */
export async function checkReme(
  io: RemeIO,
  pythonPath: string,
): Promise<RemeCheck> {
  const path = pythonPath.trim();
  if (!path || !isAbsolutePath(path)) {
    return {
      installed: false,
      version: "",
      versionOk: false,
      hasCore: false,
      cliPath: "",
      error: "Python interpreter path must be absolute",
    };
  }
  const probe = await probeReme(io, path);
  if (!probe.installed) {
    return {
      ...probe,
      versionOk: false,
      hasCore: false,
      cliPath: "",
      error: probe.error ?? "",
    };
  }
  let hasCore = false;
  try {
    const core = await io.run(path, ["-c", "import agentscope"], {
      timeoutMs: 10_000,
    });
    hasCore = !core.error && core.code === 0;
  } catch {
    hasCore = false;
  }
  return {
    ...probe,
    versionOk: remeVersionAtLeast(probe.version),
    hasCore,
    cliPath: remeCliPath(io, path),
    error: "",
  };
}
