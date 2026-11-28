import type { AgentView, ProactiveRuntimeSettings } from "./wire.ts";
export type { ProactiveRuntimeSettings } from "./wire.ts";

export interface TaskDocument {
  path: string;
  content: string;
  mtime: string;
}

export function taskSettings(status: AgentView): ProactiveRuntimeSettings {
  return {
    enabled: status.enabled,
    schedule: status.schedule,
    min_trigger_interval_seconds: status.min_trigger_interval_seconds,
    notify_windows: status.notify_windows,
  };
}

export function changedTaskSettings(
  before: ProactiveRuntimeSettings,
  after: ProactiveRuntimeSettings,
): Partial<ProactiveRuntimeSettings> {
  return Object.fromEntries(
    Object.entries(after).filter(
      ([key, value]) =>
        JSON.stringify(value) !==
        JSON.stringify(before[key as keyof ProactiveRuntimeSettings]),
    ),
  );
}

export type TaskRequest = { type: "task" } & (
  | {
      action: "create";
      agent_name: string;
      agent_type: "event" | "time";
      task_description: string;
      settings: ProactiveRuntimeSettings;
    }
  | { action: "read"; agent_name: string }
  | {
      action: "update";
      agent_name: string;
      settings: Partial<ProactiveRuntimeSettings>;
    }
  | {
      action: "save";
      agent_name: string;
      content: string;
      expected_mtime: string;
    }
);

/** The renderer selects a named operation, never a backend job or a filesystem path. */
export function taskPayload(request: TaskRequest): {
  job: string;
  body: object;
} {
  const { action, agent_name } = request;
  const fields = [
    "type",
    "action",
    "agent_name",
    ...(action === "create"
      ? ["agent_type", "task_description", "settings"]
      : action === "update"
        ? ["settings"]
        : action === "save"
          ? ["content", "expected_mtime"]
          : []),
  ];
  if (Object.keys(request).some((key) => !fields.includes(key)))
    throw new Error("Invalid task field");
  if (
    typeof agent_name !== "string" ||
    !/^[\p{L}\p{N}_-]{1,80}$/u.test(agent_name) ||
    agent_name.toLowerCase() === "remy"
  )
    throw new Error("Invalid task name");
  const body: Record<string, unknown> = { action, agent_name };
  if (action === "create" || action === "update") {
    const settings = request.settings;
    if (!settings || typeof settings !== "object")
      throw new Error("Invalid task settings");
    const keys = [
      "enabled",
      "schedule",
      "min_trigger_interval_seconds",
      "notify_windows",
    ] as const;
    if (
      Object.keys(settings).some(
        (key) => !keys.includes(key as (typeof keys)[number]),
      )
    )
      throw new Error("Invalid task setting");
    if (action === "create" && keys.some((key) => !(key in settings)))
      throw new Error("Incomplete task settings");
    for (const key of keys) if (key in settings) body[key] = settings[key];
    if (action === "update" && Object.keys(settings).length === 0)
      throw new Error("Empty task update");
    if (action === "create") {
      if (request.agent_type !== "event" && request.agent_type !== "time")
        throw new Error("Invalid task trigger type");
      if (
        (request.agent_type === "time") !==
        (typeof settings.schedule === "string")
      )
        throw new Error("Task schedule does not match its trigger type");
      body.agent_type = request.agent_type;
      if (
        typeof request.task_description !== "string" ||
        !request.task_description.trim() ||
        request.task_description.length > 100000
      )
        throw new Error("Invalid task description");
      body.task_description = request.task_description;
    }
  } else if (action === "save") {
    if (
      typeof request.content !== "string" ||
      !request.content.trim() ||
      request.content.length > 100000 ||
      typeof request.expected_mtime !== "string" ||
      !request.expected_mtime ||
      request.expected_mtime.length > 80
    )
      throw new Error("Invalid task document");
    body.content = request.content;
    body.expected_mtime = request.expected_mtime;
  } else if (action !== "read") throw new Error("Invalid task action");
  return {
    job:
      action === "read" || action === "save"
        ? "task_document"
        : "manage_proactive_runtime",
    body,
  };
}
