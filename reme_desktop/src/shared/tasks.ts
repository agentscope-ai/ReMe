export type TaskType = "proactive" | "cron";

export interface TaskSettings {
  agent_type: TaskType;
  enabled: boolean;
  cron: string | null;
  min_trigger_interval_seconds: number;
  notify_window: { start: string; end: string } | null;
}

export interface TaskStatus extends Omit<
  TaskSettings,
  "cron" | "min_trigger_interval_seconds" | "notify_window"
> {
  agent_name: string;
  task_path: string;
  cron?: string;
  min_trigger_interval_seconds?: number;
  notify_window?: TaskSettings["notify_window"];
  running: boolean;
  pending_events: number;
  in_flight_events: number;
  last_started_at?: string;
  last_completed_at: string | null;
  next_trigger_at?: string;
  error?: string;
}

export interface TaskDocument {
  path: string;
  content: string;
  mtime: string;
}

export function taskSettings(status: TaskStatus): TaskSettings {
  return {
    agent_type: status.agent_type,
    enabled: status.enabled,
    cron: status.cron ?? null,
    min_trigger_interval_seconds: status.min_trigger_interval_seconds ?? 0,
    notify_window: status.notify_window ?? null,
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
  | { action: "status" | "delete" | "read"; agent_name: string }
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
    ["remy", "tmp"].includes(agent_name.toLowerCase())
  )
    throw new Error("Invalid task name");
  const body: Record<string, unknown> = { action, agent_name };
  if (action === "create" || action === "update") {
    const settings = request.settings;
    if (!settings || typeof settings !== "object")
      throw new Error("Invalid task settings");
    const keys = [
      "agent_type",
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
  } else if (!["status", "delete", "read"].includes(action))
    throw new Error("Invalid task action");
  return {
    job:
      action === "read" || action === "save"
        ? "agent_task"
        : "manage_proactive_agent",
    body,
  };
}
