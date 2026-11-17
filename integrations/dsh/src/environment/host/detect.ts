import { win32 } from "node:path";

import { safeError } from "../../endpoint.js";
import { pathOf } from "./paths.js";
import type { RunCommand, RunResult } from "./runner.js";
import { probeReme, remeVersionAtLeast } from "./reme.js";
import type { PythonDetection, PythonEnv, PythonOrigin } from "../types.js";

/** ReMe requires Python 3.11 or newer. */
export const MINIMUM_PYTHON = "3.11";

/**
 * Ask one interpreter for its real executable, version and base prefix.
 * Kept as a single self-contained -c snippet so no helper files are needed.
 */
const VERIFY_SNIPPET =
  'import sys,json;print(json.dumps({"exe":sys.executable,"ver":"%d.%d.%d"%sys.version_info[:3],"base":sys.base_prefix}))';

export interface DetectIO {
  run: RunCommand;
  platform: NodeJS.Platform;
  env: NodeJS.ProcessEnv;
  homedir: string;
  exists(path: string): boolean;
}

export interface VerifyInfo {
  exe: string;
  version: string;
  basePrefix: string;
}

/* ------------------------------------------------------------------ */
/* Pure parsers (unit-tested against real tool output fixtures).       */
/* ------------------------------------------------------------------ */

/** Compare "major.minor[.patch]" strings numerically. */
export function versionAtLeast(
  version: string,
  minimum: string = MINIMUM_PYTHON,
): boolean {
  const actual = version.split(".").map((part) => Number.parseInt(part, 10));
  if (actual.length < 2 || actual.some((part) => !Number.isFinite(part)))
    return false;
  const wanted = minimum.split(".").map((part) => Number.parseInt(part, 10));
  for (let index = 0; index < wanted.length; index += 1) {
    const left = actual[index] ?? 0;
    const right = wanted[index] ?? 0;
    if (left !== right) return left > right;
  }
  return true;
}

/** Parse the VERIFY_SNIPPET line out of mixed stdout (banners, warnings). */
export function parseVerifyOutput(text: string): VerifyInfo | undefined {
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) continue;
    try {
      const value = JSON.parse(trimmed) as Record<string, unknown>;
      const exe = typeof value.exe === "string" ? value.exe : "";
      const version = typeof value.ver === "string" ? value.ver : "";
      const base = typeof value.base === "string" ? value.base : "";
      if (exe && version) return { exe, version, basePrefix: base };
    } catch {
      // Not JSON; try the next line.
    }
  }
  return undefined;
}

export interface RegPython {
  exe: string;
  source: string;
}

/**
 * Parse `reg query <hive> /s /ve` output for PEP 514 InstallPath defaults.
 * The default-value label is localized and arrives mojibaked when reg.exe
 * writes GBK into a UTF-8 pipe, so match the ASCII "REG_SZ" marker instead.
 */
export function parseRegQuery(text: string): RegPython[] {
  const found: RegPython[] = [];
  const lines = text.split(/\r?\n/);
  for (let index = 0; index < lines.length; index += 1) {
    const key = (lines[index] ?? "").trim();
    if (!key.startsWith("HKEY_") || !key.endsWith("\\InstallPath")) continue;
    for (
      let offset = 1;
      offset <= 3 && index + offset < lines.length;
      offset += 1
    ) {
      const line = (lines[index + offset] ?? "").trim();
      if (line.startsWith("HKEY_")) break;
      const match = /REG_SZ\s+(.+)$/i.exec(line);
      if (!match) continue;
      const dir = (match[1] ?? "").trim();
      if (!dir || /^\(.*\)$/.test(dir)) break;
      const exe = /python[^\\/]*\.exe$/i.test(dir)
        ? dir
        : `${dir.replace(/[\\/]+$/, "")}\\python.exe`;
      found.push({ exe, source: `registry ${key}` });
      break;
    }
  }
  return found;
}

/** Parse `py -0p` lines: "-V:3.12 *        D:\...\python.exe". */
export function parsePyListPaths(text: string): string[] {
  const paths: string[] = [];
  for (const line of text.split(/\r?\n/)) {
    const match = /-V:\d+(?:\.\d+)?\s+\*?\s*(\S.*)$/.exec(line);
    const exe = (match?.[1] ?? "").trim();
    if (exe && /python[^\\/]*\.exe$/i.test(exe)) paths.push(exe);
  }
  return paths;
}

/** Parse `where`/`which -a` output into trimmed, non-empty lines. */
export function parsePathList(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 2);
}

/**
 * Microsoft Store installs real interpreters under
 * WindowsApps\PythonSoftwareFoundation.*, while the alias stubs sit directly
 * in WindowsApps and only open the Store. Only the stubs must be dropped.
 */
export function isWindowsAppsAlias(exe: string): boolean {
  const normalized = exe.replaceAll("/", "\\").toLowerCase();
  const cut = normalized.lastIndexOf("\\");
  if (cut < 0) return false;
  const parent = normalized.slice(0, cut);
  const parentName = parent.slice(parent.lastIndexOf("\\") + 1);
  return parentName === "windowsapps";
}

function isWindowsStorePackage(exe: string): boolean {
  return /\\windowsapps\\pythonsoftwarefoundation\./i.test(
    exe.replaceAll("/", "\\"),
  );
}

export function parseCondaEnvs(text: string): string[] {
  try {
    const value = JSON.parse(text) as { envs?: unknown };
    if (!Array.isArray(value.envs)) return [];
    return value.envs.filter(
      (item): item is string => typeof item === "string" && item.length > 0,
    );
  } catch {
    return [];
  }
}

export function parseCondaRoot(text: string): string {
  try {
    const value = JSON.parse(text) as { root_prefix?: unknown };
    return typeof value.root_prefix === "string" ? value.root_prefix : "";
  } catch {
    return "";
  }
}

/* ------------------------------------------------------------------ */
/* Conda command runner: prefer the real conda.exe, fall back to .bat. */
/* ------------------------------------------------------------------ */

export type CondaRunner = (
  args: string[],
  timeoutMs?: number,
) => Promise<RunResult>;

/** Derive the conda installation root from any conda executable path. */
export function condaRootFromPath(condaPath: string): string {
  // The path may be written for another platform than the host, so its flavor
  // decides the semantics: dirname must not depend on where tests run.
  const path = pathOf(condaPath);
  const dir = path.dirname(condaPath);
  const lower = dir.toLowerCase();
  if (lower.endsWith("\\library\\bin") || lower.endsWith("/library/bin"))
    return path.dirname(path.dirname(dir));
  if (lower.endsWith("\\scripts") || lower.endsWith("/scripts"))
    return path.dirname(dir);
  if (lower.endsWith("\\condabin") || lower.endsWith("/condabin"))
    return path.dirname(dir);
  if (lower.endsWith("\\bin") || lower.endsWith("/bin"))
    return path.dirname(dir);
  return "";
}

/**
 * Windows ships conda as .bat files that Node cannot spawn without a shell;
 * run the real Scripts\conda.exe when present, otherwise wrap the .bat in
 * cmd.exe with verbatim argv. The path comes from discovery, never user input.
 */
export function makeCondaRunner(
  run: RunCommand,
  condaPath: string,
  windows: boolean,
  exists: (path: string) => boolean,
): CondaRunner {
  const root = condaRootFromPath(condaPath);
  const exeCandidates: string[] = [];
  if (root && windows)
    exeCandidates.push(win32.join(root, "Scripts", "conda.exe"));
  if (!/\.(bat|cmd)$/i.test(condaPath)) exeCandidates.push(condaPath);
  return async (args, timeoutMs = 15_000) => {
    for (const candidate of exeCandidates) {
      if (!exists(candidate)) continue;
      const result = await run(candidate, args, { timeoutMs });
      if (!result.error) return result;
    }
    if (windows)
      return run(
        "cmd.exe",
        ["/d", "/s", "/c", `"${condaPath}" ${args.join(" ")}`],
        { timeoutMs, windowsVerbatimArguments: true },
      );
    return run(condaPath, args, { timeoutMs });
  };
}

/**
 * Unix interpreters that a `which -a python3` can miss when the shell has a
 * trimmed PATH: the system install, Homebrew on both Apple Silicon and Intel,
 * and pyenv/local shims. Windows has `where` and the registry for this.
 */
function commonPythonPaths(io: DetectIO, windows: boolean): string[] {
  if (windows) return [];
  const path = pathOf(io.homedir);
  return [
    "/usr/bin/python3",
    "/usr/bin/python",
    "/usr/local/bin/python3",
    "/opt/homebrew/bin/python3",
    path.join(io.homedir, ".pyenv", "shims", "python3"),
    path.join(io.homedir, ".local", "bin", "python3"),
  ];
}

function commonCondaPaths(io: DetectIO, windows: boolean): string[] {
  const programData = io.env["ProgramData"] ?? "";
  const path = windows ? win32 : pathOf(io.homedir);
  const candidates = windows
    ? [
        path.join(io.homedir, "miniconda3", "Scripts", "conda.exe"),
        path.join(io.homedir, "anaconda3", "Scripts", "conda.exe"),
        path.join(programData, "miniconda3", "Scripts", "conda.exe"),
        path.join(programData, "anaconda3", "Scripts", "conda.exe"),
      ]
    : [
        path.join(io.homedir, "miniconda3", "bin", "conda"),
        path.join(io.homedir, "anaconda3", "bin", "conda"),
        path.join(io.homedir, "miniforge3", "bin", "conda"),
        "/opt/miniconda3/bin/conda",
        "/opt/anaconda3/bin/conda",
        "/opt/homebrew/Caskroom/miniconda/base/bin/conda",
      ];
  return candidates.filter((candidate) => path.isAbsolute(candidate));
}

/* ------------------------------------------------------------------ */
/* Orchestration.                                                      */
/* ------------------------------------------------------------------ */

interface Candidate {
  exe: string;
  sources: Set<string>;
}

function normalizeExe(exe: string, windows: boolean): string {
  const trimmed = exe.trim();
  if (!trimmed) return "";
  // Unify separators for the simulated platform: resolve() follows the host,
  // but detection must behave identically on every host.
  const unified = windows
    ? trimmed.replaceAll("/", "\\")
    : trimmed.replaceAll("\\", "/");
  return windows ? unified.toLowerCase() : unified;
}

function addCandidate(
  candidates: Map<string, Candidate>,
  exe: string,
  source: string,
  windows: boolean,
): void {
  const key = normalizeExe(exe, windows);
  if (!key) return;
  const existing = candidates.get(key);
  if (existing) existing.sources.add(source);
  else candidates.set(key, { exe: exe.trim(), sources: new Set([source]) });
}

function pathContains(dir: string, exe: string, windows: boolean): boolean {
  const path = windows ? win32 : pathOf(exe);
  const base = normalizeExe(path.resolve(dir), windows);
  const full = normalizeExe(path.resolve(exe), windows);
  return full.startsWith(`${base}${windows ? "\\" : "/"}`);
}

/** prefix of an interpreter install from its executable location. */
function pythonPrefix(exe: string, windows: boolean): string {
  const path = windows ? win32 : pathOf(exe);
  const parent = path.dirname(exe);
  return windows || path.basename(parent) !== "bin"
    ? parent
    : path.dirname(parent);
}

function isVenv(info: VerifyInfo, windows: boolean): boolean {
  if (!info.basePrefix) return false;
  return (
    normalizeExe(pythonPrefix(info.exe, windows), windows) !==
    normalizeExe(info.basePrefix, windows)
  );
}

const KIND_RANK: Record<PythonOrigin["kind"], number> = {
  conda: 0,
  "python.org": 1,
  "windows-store": 2,
  venv: 3,
  system: 4,
  unknown: 5,
};

function classifyOrigin(
  info: VerifyInfo,
  candidate: Candidate,
  conda: { root: string; envs: string[] },
  windows: boolean,
): PythonOrigin {
  // Named envs live under the root too, so they must be matched first.
  for (const envPath of conda.envs) {
    if (envPath === conda.root) continue;
    if (pathContains(envPath, info.exe, windows))
      return {
        kind: "conda",
        envName: (windows ? win32 : pathOf(envPath)).basename(envPath),
        root: envPath,
      };
  }
  if (conda.root && pathContains(conda.root, info.exe, windows))
    return { kind: "conda", envName: "base", root: conda.root };
  if (isWindowsStorePackage(info.exe)) return { kind: "windows-store" };
  if (isVenv(info, windows))
    return {
      kind: "venv",
      baseExe: windows
        ? win32.join(info.basePrefix, "python.exe")
        : pathOf(info.basePrefix).join(info.basePrefix, "bin", "python3"),
    };
  if ([...candidate.sources].some((source) => source.startsWith("registry")))
    return { kind: "python.org" };
  if (
    candidate.sources.has("PATH") ||
    candidate.sources.has("which") ||
    candidate.sources.has("well-known")
  )
    return { kind: "system" };
  return { kind: "unknown" };
}

function compareEnvs(a: PythonEnv, b: PythonEnv): number {
  if (a.ok !== b.ok) return a.ok ? -1 : 1;
  const rank = KIND_RANK[a.origin.kind] - KIND_RANK[b.origin.kind];
  if (rank !== 0) return rank;
  const nameA = a.origin.kind === "conda" ? a.origin.envName : "";
  const nameB = b.origin.kind === "conda" ? b.origin.envName : "";
  if (nameA !== nameB) return nameA.localeCompare(nameB);
  return a.exe.localeCompare(b.exe);
}

/** Prefer a named conda env with a usable reme, then any reme env. */
export function suggestInterpreter(envs: PythonEnv[]): string {
  const usable = envs.filter((env) => env.ok);
  // An installed-but-outdated reme does not make an interpreter ready.
  const withReme = usable.filter(
    (env) =>
      env.reme?.installed === true && remeVersionAtLeast(env.reme.version),
  );
  const named = (list: PythonEnv[]): PythonEnv | undefined =>
    list.find(
      (env) => env.origin.kind === "conda" && env.origin.envName !== "base",
    );
  return (
    (named(withReme) ?? withReme[0] ?? named(usable) ?? usable[0])?.exe ?? ""
  );
}

/** Probe reme-ai in every interpreter so the list shows setup state. */
async function enrichWithReme(envs: PythonEnv[], io: DetectIO): Promise<void> {
  const queue = [...envs];
  const worker = async (): Promise<void> => {
    for (;;) {
      const env = queue.shift();
      if (!env) return;
      env.reme = await probeReme(io, env.exe);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(6, envs.length) }, () => worker()),
  );
}

async function verifyCandidates(
  candidates: Map<string, Candidate>,
  io: DetectIO,
): Promise<Map<string, VerifyInfo>> {
  const verified = new Map<string, VerifyInfo>();
  const queue = [...candidates.entries()];
  const worker = async (): Promise<void> => {
    for (;;) {
      const entry = queue.shift();
      if (!entry) return;
      const [key, candidate] = entry;
      let result: RunResult;
      try {
        result = await io.run(candidate.exe, ["-c", VERIFY_SNIPPET], {
          timeoutMs: 8_000,
        });
      } catch {
        continue;
      }
      if (result.error || result.code !== 0) continue;
      const info = parseVerifyOutput(result.stdout);
      if (!info) continue;
      // Prefer the interpreter's own sys.executable (resolves symlinks/shims).
      verified.set(key, { ...info, exe: info.exe || candidate.exe });
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(6, queue.length) }, () => worker()),
  );
  return verified;
}

/**
 * Discover local Python interpreters, verify each one by running it, and
 * classify origins. Discovery failures are collected; they never reject.
 */
export async function detectPythonEnvs(io: DetectIO): Promise<PythonDetection> {
  const windows = io.platform === "win32";
  const errors: string[] = [];
  const candidates = new Map<string, Candidate>();
  let condaRoot = "";
  let condaEnvs: string[] = [];

  const steps: Promise<void>[] = [
    // PEP 514 registry (Windows; Miniconda registers itself here too).
    (async () => {
      if (!windows) return;
      const hives = [
        "HKCU\\Software\\Python",
        "HKLM\\Software\\Python",
        "HKLM\\Software\\WOW6432Node\\Python",
      ];
      for (const hive of hives) {
        const result = await io.run("reg.exe", ["query", hive, "/s", "/ve"], {
          timeoutMs: 5_000,
        });
        if (result.error) continue;
        for (const { exe, source } of parseRegQuery(result.stdout))
          addCandidate(candidates, exe, source, windows);
      }
    })(),
    // py launcher (Windows only; usually absent without a python.org install).
    (async () => {
      if (!windows) return;
      const result = await io.run("py.exe", ["-0p"], { timeoutMs: 5_000 });
      if (result.error) return;
      for (const exe of parsePyListPaths(result.stdout))
        addCandidate(candidates, exe, "py launcher", windows);
    })(),
    // PATH / which.
    (async () => {
      const first = windows
        ? await io.run("where.exe", ["python"], { timeoutMs: 5_000 })
        : await io.run("which", ["-a", "python3"], { timeoutMs: 5_000 });
      // A non-zero exit just means "no match"; a spawn/timeout error is a real
      // diagnostic and must not disappear from the report.
      if (first.error) errors.push(`python: ${safeError(first.error)}`);
      let stdout = first.stdout;
      if (!windows) {
        const extra = await io.run("which", ["-a", "python"], {
          timeoutMs: 5_000,
        });
        if (extra.error) errors.push(`python: ${safeError(extra.error)}`);
        stdout = `${stdout}\n${extra.stdout}`;
      }
      for (const exe of parsePathList(stdout)) {
        if (isWindowsAppsAlias(exe)) continue;
        addCandidate(candidates, exe, windows ? "PATH" : "which", windows);
      }
      // Well-known Unix install locations, independent of the Host PATH.
      for (const exe of commonPythonPaths(io, windows))
        if (io.exists(exe))
          addCandidate(candidates, exe, "well-known", windows);
    })(),
    // conda: PATH first, then well-known install roots.
    (async () => {
      const located = windows
        ? parsePathList(
            (await io.run("where.exe", ["conda"], { timeoutMs: 5_000 })).stdout,
          )
        : parsePathList(
            (await io.run("which", ["-a", "conda"], { timeoutMs: 5_000 }))
              .stdout,
          );
      for (const path of commonCondaPaths(io, windows))
        if (io.exists(path)) located.push(path);
      if (!located.length) return;
      const runConda = makeCondaRunner(
        io.run,
        located[0] ?? "",
        windows,
        io.exists,
      );
      const info = await runConda(["info", "--json"]);
      if (info.error)
        errors.push(
          `conda: discovered but not runnable (${safeError(info.error)})`,
        );
      condaRoot = info.error ? "" : parseCondaRoot(info.stdout);
      const listing = await runConda(["env", "list", "--json"]);
      condaEnvs = listing.error ? [] : parseCondaEnvs(listing.stdout);
      if (!condaRoot && condaEnvs.length) condaRoot = condaEnvs[0] ?? "";
      for (const envPath of condaEnvs) {
        addCandidate(
          candidates,
          windows
            ? win32.join(envPath, "python.exe")
            : pathOf(envPath).join(envPath, "bin", "python3"),
          "conda env list",
          windows,
        );
      }
    })().catch((error: unknown) => {
      errors.push(`conda: ${safeError(error)}`);
    }),
  ];

  await Promise.allSettled(steps);

  const verified = await verifyCandidates(candidates, io);
  // Candidates can resolve to the same real executable (aliases, shims);
  // merge their discovery sources so the list stays duplicate-free.
  const merged = new Map<string, PythonEnv>();
  for (const [key, candidate] of candidates) {
    const info = verified.get(key);
    if (!info) continue;
    const exe = info.exe || candidate.exe;
    const dedupeKey = normalizeExe(exe, windows);
    const existing = merged.get(dedupeKey);
    if (existing) {
      existing.source = [
        ...new Set([...existing.source.split("; "), ...candidate.sources]),
      ]
        .sort()
        .join("; ");
      continue;
    }
    merged.set(dedupeKey, {
      exe,
      version: info.version,
      ok: versionAtLeast(info.version),
      origin: classifyOrigin(
        info,
        candidate,
        { root: condaRoot, envs: condaEnvs },
        windows,
      ),
      source: [...candidate.sources].sort().join("; "),
    });
  }
  const envs = [...merged.values()];
  await enrichWithReme(envs, io);
  envs.sort(compareEnvs);
  return {
    platform: io.platform,
    scannedAt: new Date().toISOString(),
    envs,
    suggested: suggestInterpreter(envs),
    condaRoot,
    errors,
  };
}
