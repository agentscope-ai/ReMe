/**
 * Canonical identity of an endpoint, shared by the Host (service records) and
 * the tab (matching an answer to the URL on screen).
 *
 * Loopback aliases and default ports compare equal, so saving
 * `localhost:2333` does not make the service this plugin started on
 * `127.0.0.1:2333` look foreign. This module stays browser-safe: no Node
 * built-ins, because the client imports it too.
 */

/**
 * The origin and root-normalized path of an endpoint, when it is parseable.
 *
 * Loopback aliases and default ports are already folded here, so both keys
 * below compare the same spelling of the same endpoint.
 */
function endpointParts(
  endpoint: string,
): { origin: string; path: string } | undefined {
  let url: URL;
  try {
    url = new URL(endpoint.trim());
  } catch {
    return undefined;
  }
  const hostname = url.hostname.toLowerCase();
  const host = hostname === "localhost" ? "127.0.0.1" : hostname;
  const port = url.port || (url.protocol === "https:" ? "443" : "80");
  return {
    origin: `${url.protocol}//${host}:${port}`,
    path: url.pathname.replace(/\/+$/, ""),
  };
}

export function serviceEndpointKey(endpoint: string): string {
  const parts = endpointParts(endpoint);
  return parts ? `${parts.origin}${parts.path}` : endpoint.trim();
}

/**
 * Canonical identity of an endpoint's origin, ignoring its path.
 *
 * The service launch configures only host and port, so two endpoints on the
 * same origin name the same local process even when their paths differ. A
 * record left by a version that still launched a prefixed endpoint (for
 * example `/reme`) therefore still describes the service a start on the
 * origin root has to replace.
 */
export function serviceOriginKey(endpoint: string): string {
  const parts = endpointParts(endpoint);
  return parts ? parts.origin : endpoint.trim();
}
