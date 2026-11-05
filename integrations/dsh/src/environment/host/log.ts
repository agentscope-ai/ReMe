import {
  mkdirSync,
  openSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";

/** Rotate a log once it grows past this size; only one backup is kept. */
export const LOG_ROTATE_BYTES = 5 * 1024 * 1024;
/** Number of per-job install logs kept on disk. */
export const INSTALL_LOG_KEEP = 20;

/**
 * Open a log for appending, rotating an oversized one to `<path>.1` first.
 * Install and service logs otherwise grow without bound because both are
 * append-only and outlive many sessions.
 */
export function openAppendLog(
  path: string,
  maxBytes = LOG_ROTATE_BYTES,
): number {
  mkdirSync(dirname(path), { recursive: true });
  try {
    if (statSync(path).size > maxBytes) renameSync(path, `${path}.1`);
  } catch {
    // A missing log has nothing to rotate; a locked one (Windows, while the
    // service still writes it) is left alone rather than failing the launch.
  }
  return openSync(path, "a");
}

/**
 * Keep only the newest `keep` per-job logs. Each install writes a new file, so
 * size-based rotation alone never bounds the directory.
 *
 * `active` are logs a running job is still writing; they are never removed.
 * Each removal is attempted on its own, because one locked file (Windows) must
 * not abort the rest of the housekeeping.
 */
export function pruneInstallLogs(
  directory: string,
  keep = INSTALL_LOG_KEEP,
  active: readonly string[] = [],
): void {
  try {
    const keepSet = new Set(active.map((path) => basename(path)));
    const entries = readdirSync(directory).filter(
      (name) => name.startsWith("install-") && name.endsWith(".log"),
    );
    if (entries.length <= keep) return;
    const byAge = entries
      .map((name) => {
        try {
          return { name, mtime: statSync(join(directory, name)).mtimeMs };
        } catch {
          return { name, mtime: 0 };
        }
      })
      .sort((left, right) => right.mtime - left.mtime);
    for (const old of byAge.slice(keep)) {
      if (keepSet.has(old.name)) continue;
      try {
        rmSync(join(directory, old.name), { force: true });
      } catch {
        // A log another job still holds open is retried next time.
      }
    }
  } catch {
    // Pruning is housekeeping; a missing directory is not an error.
  }
}
