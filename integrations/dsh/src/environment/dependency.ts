/**
 * What one interpreter can and cannot run.
 *
 * The dependency card, the hero and the service card all have to answer the
 * same question about the selected interpreter, and the Host answers it again
 * when a start is requested. The answers live here so the four cannot drift
 * apart: a button that is offered for an interpreter the Host will refuse
 * only produces an error, and a button that is hidden for one the Host would
 * accept takes a capability away.
 *
 * This module stays browser-safe: the client imports it too.
 */
import type { RemeCheck } from "./types.js";

/**
 * Whether the interpreter can host the ReMe service.
 *
 * This is the Host's own start gate: `reme-ai` must be installed and satisfy
 * the minimum version, because the Host refuses a start otherwise with
 * `not-installed` or `version-too-old`.
 */
export function canHostService(check: RemeCheck | undefined): boolean {
  return check?.installed === true && check.versionOk === true;
}

/**
 * Whether the interpreter offers the whole feature set.
 *
 * The `[core]` extras provide automatic memory and automatic consolidation, so
 * an interpreter without them runs the service but not those two jobs. The
 * hero reports this state; a start is still offered, because the Host accepts
 * one and search still works.
 */
export function hasCoreFeatures(check: RemeCheck | undefined): boolean {
  return canHostService(check) && check?.hasCore === true;
}
