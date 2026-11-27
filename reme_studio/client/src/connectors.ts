import type { ConnectorView } from "./wire.ts";
export type { ConnectorView } from "./wire.ts";

export type ConnectorRequest = { type: "connector" } & (
  | { action: "list" }
  | {
      action: "create";
      name: string;
      runtime_names: string[];
      binding: { url: string };
    }
  | {
      action: "update";
      source_id: string;
      name?: string;
      enabled?: boolean;
      runtime_names?: string[];
    }
);

/** A narrow host request; icons and presentation preferences never enter the payload. */
export function connectorPayload(request: ConnectorRequest): object {
  const { action } = request;
  const fields = [
    "type",
    "action",
    ...(action === "list"
      ? []
      : action === "create"
        ? ["name", "runtime_names", "binding"]
        : ["source_id", "name", "enabled", "runtime_names"]),
  ];
  if (
    !["list", "create", "update"].includes(action) ||
    Object.keys(request).some((key) => !fields.includes(key))
  )
    throw new Error("Invalid source operation");
  if (action === "list") return { action };
  const result: Record<string, unknown> = { action };
  if (action === "create") {
    const url = new URL(request.binding.url);
    if (!["http:", "https:"].includes(url.protocol))
      throw new Error("Invalid source URL");
    result.binding = { url: url.href };
  } else {
    if (
      !/^(?:source_[0-9a-f]{32}|preset:[\p{L}\p{N}_-]{1,80})$/u.test(
        request.source_id,
      )
    )
      throw new Error("Invalid source identity");
    result.source_id = request.source_id;
    if (request.enabled !== undefined) {
      if (typeof request.enabled !== "boolean")
        throw new Error("Invalid source enabled flag");
      result.enabled = request.enabled;
    }
  }
  if ("name" in request && request.name !== undefined) {
    if (
      typeof request.name !== "string" ||
      !request.name.trim() ||
      request.name.length > 80
    )
      throw new Error("Invalid source name");
    result.name = request.name.trim();
  }
  if ("runtime_names" in request && request.runtime_names !== undefined) {
    const names = request.runtime_names;
    if (
      !Array.isArray(names) ||
      (action === "create" && !names.length) ||
      names.some(
        (name) =>
          typeof name !== "string" || !/^[\p{L}\p{N}_-]{1,80}$/u.test(name),
      ) ||
      new Set(names.map((name) => name.toLowerCase())).size !== names.length
    )
      throw new Error("Invalid source targets");
    result.runtime_names = names;
  }
  if (action === "create" && (!result.name || !result.runtime_names))
    throw new Error("Incomplete source definition");
  if (action === "update" && Object.keys(result).length === 2)
    throw new Error("Empty source update");
  return result;
}

export type ConnectorList = { items: ConnectorView[]; total: number };
