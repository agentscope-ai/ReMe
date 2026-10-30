/**
 * The endpoint rules the Host and the tab must agree on.
 *
 * The plugin launches `reme start` with `service.host=127.0.0.1` and a plain
 * HTTP service, so only a loopback HTTP endpoint can ever become a managed
 * local service. The tab decides whether to show the local steps at all and
 * the Host decides whether it may act; both read the same rules here, so the
 * browser cannot offer a local service the Host would refuse.
 *
 * This module stays browser-safe: the client imports it too.
 */

/** The host and port of a manageable local endpoint. */
export interface LocalEndpoint {
  host: string;
  port: number;
}

/**
 * The host and port of a manageable local endpoint.
 *
 * Throws when the endpoint is not one, so the caller can report why. An
 * https loopback URL is refused: the service this plugin launches is plain
 * HTTP, so an https URL could never become healthy.
 */
export function parseLocalEndpoint(endpoint: string): LocalEndpoint {
  let url: URL;
  try {
    url = new URL(endpoint.trim());
  } catch {
    throw new TypeError("Invalid ReMe endpoint");
  }
  if (url.protocol !== "http:")
    throw new TypeError(
      "Local service management requires an http:// endpoint",
    );
  // The service is launched with `service.host=127.0.0.1`; accepting an IPv6
  // loopback would spawn a process that can never answer that endpoint.
  if (!["127.0.0.1", "localhost"].includes(url.hostname))
    throw new TypeError(
      "Local service management requires an IPv4 loopback endpoint",
    );
  const port = Number(url.port || "80");
  if (!Number.isInteger(port) || port < 1 || port > 65535)
    throw new TypeError("Invalid ReMe endpoint port");
  return { host: "127.0.0.1", port };
}

/**
 * Whether the endpoint names a service this plugin can manage locally.
 *
 * A remote endpoint is still health-checked, but its local steps are hidden:
 * the browser and the Host may be different machines.
 */
export function isLocalEndpoint(endpoint: string): boolean {
  try {
    parseLocalEndpoint(endpoint);
    return true;
  } catch {
    return false;
  }
}
