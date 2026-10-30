/**
 * Canonical identity of an endpoint, shared by the Host (service records) and
 * the tab (matching an answer to the URL on screen).
 *
 * Loopback aliases and default ports compare equal, so saving
 * `localhost:2333` does not make the service this plugin started on
 * `127.0.0.1:2333` look foreign. This module stays browser-safe: no Node
 * built-ins, because the client imports it too.
 */
export function serviceEndpointKey(endpoint: string): string {
  try {
    const url = new URL(endpoint.trim());
    const hostname = url.hostname.toLowerCase();
    const host = hostname === "localhost" ? "127.0.0.1" : hostname;
    const port = url.port || (url.protocol === "https:" ? "443" : "80");
    const path = url.pathname.replace(/\/+$/, "");
    return `${url.protocol}//${host}:${port}${path}`;
  } catch {
    return endpoint.trim();
  }
}
