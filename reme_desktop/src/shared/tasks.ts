import type { AgentView, TaskSettings } from "./wire.ts";
export type { TaskSettings } from "./wire.ts";

export type TaskType = "proactive" | "cron";

export interface TaskDocument {
  path: string;
  content: string;
  mtime: string;
}

export function taskSettings(status: AgentView): TaskSettings {
  return {
    enabled: status.enabled,
    cron: status.cron,
    min_trigger_interval_seconds: status.min_trigger_interval_seconds,
    notify_window: status.notify_window,
  };
}

export function changedTaskSettings(
  before: TaskSettings,
  after: TaskSettings,
): Partial<TaskSettings> {
  return Object.fromEntries(
    Object.entries(after).filter(
      ([key, value]) =>
        JSON.stringify(value) !==
        JSON.stringify(before[key as keyof TaskSettings]),
    ),
  );
}

export type TaskRequest = { type: "task" } & (
  | {
      action: "create";
      agent_name: string;
      task_description: string;
      settings: TaskSettings;
    }
  | { action: "delete" | "read"; agent_name: string }
  | { action: "update"; agent_name: string; settings: Partial<TaskSettings> }
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
      ? ["task_description", "settings"]
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
      "cron",
      "min_trigger_interval_seconds",
      "notify_window",
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
  } else if (!["delete", "read"].includes(action))
    throw new Error("Invalid task action");
  return {
    job:
      action === "read" || action === "save"
        ? "agent_task"
        : "manage_proactive_agent",
    body,
  };
}
