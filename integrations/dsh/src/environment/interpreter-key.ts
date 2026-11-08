/**
 * Canonical identity of an interpreter path, shared by the Host (install jobs)
 * and the tab (matching a job to the interpreter on screen).
 *
 * Windows paths fold case and separators, because Windows does; POSIX paths
 * stay case-sensitive. This module stays browser-safe: no Node built-ins,
 * because the client imports it too.
 */

/** True when a path is written the Windows way: drive letter, UNC or backslash. */
export function isWindowsPath(value: string): boolean {
  return (
    /^[A-Za-z]:[\\/]/.test(value) ||
    value.startsWith("\\\\") ||
    value.includes("\\")
  );
}

/**
 * The key two interpreter paths are compared by.
 *
 * The path flavor decides the comparison: a Windows path is written with
 * forward slashes and lower case, a POSIX path keeps its own spelling,
 * because it is case-sensitive there.
 */
export function interpreterKey(path: string): string {
  const value = path.trim();
  if (!isWindowsPath(value)) return value;
  return value.replaceAll("\\", "/").toLowerCase();
}
