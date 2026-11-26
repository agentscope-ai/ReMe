import { translator, type Language } from "./i18n.ts";

export const DEFAULT_PROJECT = "随笔";

export function normalizeProject(value: string): string {
  return value.trim() || DEFAULT_PROJECT;
}

/** Translate the built-in project without changing user-owned project names. */
export function projectLabel(value: string, language: Language): string {
  const project = normalizeProject(value);
  return project === DEFAULT_PROJECT
    ? translator(language)("notebookDefaultProject")
    : project;
}

/** A localized default entered in a project field resolves to its stable value. */
export function projectValue(value: string, language: Language): string {
  return value.trim() === translator(language)("notebookDefaultProject")
    ? DEFAULT_PROJECT
    : value;
}

export type NotebookKind = "todo" | "idea";
const todoStatuses = ["pending", "in_progress", "closed"] as const;
export type TodoStatus = (typeof todoStatuses)[number];

interface NotebookRecord {
  id: string;
  project: string;
  text: string;
  created_at?: string | null;
  closed_at?: string | null;
}

export type Idea = NotebookRecord;

export interface Todo extends NotebookRecord {
  status: TodoStatus;
  progress: { time: string; text: string }[];
  progress_count?: number;
  due_date?: string | null;
  source_idea?: string | null;
  close_note?: string | null;
}

export type NotebookItem = Todo | Idea;
export const isTodo = (item: NotebookItem): item is Todo => "status" in item;

export function splitNotebookText(text: string) {
  const [title, ...lines] = text.trim().split(/\r?\n/);
  return { title, body: lines.join("\n").trim() };
}

interface ListOptions {
  action: "list";
  project?: string;
  query?: string;
  scope?: "active" | "archive" | "all";
  offset?: number;
  limit?: number;
}

export interface NotebookProjects {
  items: { project: string; todo_count: number; idea_count: number }[];
  total: number;
}

interface ContentFields {
  project?: string;
  text?: string;
}

interface TodoFields extends ContentFields {
  status?: TodoStatus;
  due_date?: string | null;
  source_idea?: string | null;
  progress_text?: string;
  close_note?: string | null;
}

export type NotebookRequest =
  | ({ type: "todo" } & (
      | (ListOptions & {
          status?: TodoStatus;
          source_idea?: string | null;
          order?: "due" | "created";
        })
      | { action: "list_projects"; scope?: ListOptions["scope"] }
      | { action: "get"; id: string }
      | ({ action: "create" } & TodoFields)
      | ({ action: "update"; id: string } & TodoFields)
    ))
  | ({ type: "idea" } & (
      | ListOptions
      | { action: "get" | "archive" | "restore"; id: string }
      | ({ action: "create" } & ContentFields)
      | ({ action: "update"; id: string } & ContentFields)
    ));

/** Match the backend's action boundaries before accepting renderer IPC. */
export function notebookPayload(
  value: NotebookRequest,
): Record<string, unknown> {
  const actions =
    value.type === "todo"
      ? ["create", "list", "list_projects", "get", "update"]
      : ["create", "list", "get", "update", "archive", "restore"];
  if (!actions.includes(value.action))
    throw new Error("Invalid notebook action");
  const fields =
    value.action === "list"
      ? [
          "project",
          "query",
          "scope",
          "offset",
          "limit",
          ...(value.type === "todo" ? ["status", "source_idea", "order"] : []),
        ]
      : value.action === "list_projects"
        ? ["scope"]
        : value.action === "create" || value.action === "update"
          ? [
              "project",
              "text",
              ...(value.action === "update" ? ["id"] : []),
              ...(value.type === "todo"
                ? [
                    "status",
                    "due_date",
                    "source_idea",
                    "progress_text",
                    "close_note",
                  ]
                : []),
            ]
          : ["id"];
  const body: Record<string, unknown> = { action: value.action };
  if (value.action !== "list_projects") body.kind = value.type;
  for (const [key, field] of Object.entries(value)) {
    if (key === "type" || key === "action") continue;
    if (!fields.includes(key)) throw new Error("Invalid notebook field");
    if (key === "offset" || key === "limit") {
      if (
        !Number.isSafeInteger(field) ||
        (field as number) < (key === "limit" ? 1 : 0) ||
        (key === "limit" && (field as number) > 100)
      )
        throw new Error("Invalid notebook pagination");
    } else if (
      (key === "due_date" || key === "source_idea" || key === "close_note") &&
      field === null
    ) {
      body[key] = null;
      continue;
    } else if (
      typeof field !== "string" ||
      field.length > 100000 ||
      (["id", "text", "source_idea", "progress_text", "close_note"].includes(
        key,
      ) &&
        !field.trim())
    ) {
      throw new Error("Invalid notebook value");
    }
    if (key === "status" && !todoStatuses.includes(field as TodoStatus))
      throw new Error("Invalid notebook status");
    if (key === "order" && field !== "due" && field !== "created")
      throw new Error("Invalid notebook order");
    if (
      key === "scope" &&
      !["active", "archive", "all"].includes(field as string)
    )
      throw new Error("Invalid notebook scope");
    body[key] = key === "project" ? normalizeProject(field as string) : field;
  }
  if (
    ["get", "update", "archive", "restore"].includes(value.action) &&
    !body.id
  )
    throw new Error("Notebook id is required");
  if (value.action === "create" && !body.text && !body.source_idea)
    throw new Error("Notebook text is required");
  if (value.action === "update" && Object.keys(body).length < 4)
    throw new Error("Notebook update is empty");
  if ("close_note" in body && body.status !== "closed")
    throw new Error("A close note requires a closed todo");
  return body;
}
