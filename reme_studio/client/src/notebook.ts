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
export type TodoStatus = "open" | "partial" | "done" | "cancelled";

export interface Idea {
  id: string;
  project: string;
  text: string;
  closed_at?: string | null;
}

export interface Todo extends Idea {
  status: TodoStatus;
  progress: { time: string; text: string }[];
  progress_count?: number;
  due_date?: string | null;
  source_idea?: string | null;
}

export type NotebookItem = Todo | Idea;
export const isTodo = (item: NotebookItem): item is Todo => "status" in item;

interface ListOptions {
  action: "list";
  project?: string;
  query?: string;
  scope?: "active" | "archive" | "all";
  offset?: number;
  limit?: number;
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
}

export type NotebookRequest =
  | ({ type: "todo" } & (
      | (ListOptions & { status?: TodoStatus; source_idea?: string | null })
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
      ? ["create", "list", "get", "update"]
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
          ...(value.type === "todo" ? ["status", "source_idea"] : []),
        ]
      : value.action === "create" || value.action === "update"
        ? [
            "project",
            "text",
            ...(value.action === "update" ? ["id"] : []),
            ...(value.type === "todo"
              ? ["status", "due_date", "source_idea", "progress_text"]
              : []),
          ]
        : ["id"];
  const body: Record<string, unknown> = { action: value.action };
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
      (key === "due_date" || key === "source_idea") &&
      field === null
    ) {
      body[key] = null;
      continue;
    } else if (
      typeof field !== "string" ||
      field.length > 100000 ||
      (["id", "text", "source_idea", "progress_text"].includes(key) &&
        !field.trim())
    ) {
      throw new Error("Invalid notebook value");
    }
    if (
      key === "status" &&
      !["open", "partial", "done", "cancelled"].includes(field as string)
    )
      throw new Error("Invalid notebook status");
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
  if (value.action === "update" && Object.keys(body).length < 3)
    throw new Error("Notebook update is empty");
  return body;
}
