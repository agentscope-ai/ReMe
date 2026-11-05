/** Shared, browser-safe readiness rule for the Host and setup cards. */
import type { RemeCheck } from "./types.js";

/**
 * Supported ReMe releases import AgentScope while loading the CLI, so a bare
 * installation cannot start the service. Install `reme-ai[core]` first.
 */
export function canHostService(check: RemeCheck | undefined): boolean {
  return (
    check?.installed === true &&
    check.versionOk === true &&
    check.hasCore === true
  );
}
