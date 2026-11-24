import type { JobResponse, WorkspaceRequest } from "./protocol.ts";
import { UiError } from "./i18n.ts";
import type { ReMeClient } from "./reme-client.ts";
import { notebookPayload } from "./notebook.ts";
import { taskPayload } from "./tasks.ts";

function relativePath(value: unknown): string {
  if (
    typeof value !== "string" ||
    value.length > 4096 ||
    value.startsWith("/") ||
    value.startsWith("~") ||
    value.includes("\\") ||
    value.includes("\0") ||
    value.split("/").includes("..") ||
    /^[a-z]+:/i.test(value)
  )
    throw new UiError("invalidWorkspaceRequest");
  return value;
}

/** Expose workspace browsing, reminder details and the two notebook jobs. */
export async function requestWorkspace(
  client: ReMeClient,
  request: WorkspaceRequest,
  signal: AbortSignal,
): Promise<JobResponse> {
  if (!request || typeof request.type !== "string")
    throw new UiError("invalidWorkspaceRequest");
  const page = "page" in request ? request.page : 1;
  if (!Number.isSafeInteger(page) || page < 1)
    throw new UiError("invalidWorkspaceRequest");
  if (
    "tags" in request &&
    (!Array.isArray(request.tags) ||
      request.tags.length > 100 ||
      request.tags.some((tag) => typeof tag !== "string" || tag.length > 256))
  )
    throw new UiError("invalidWorkspaceRequest");
  let job: string;
  let body: object;
  switch (request.type) {
    case "task": {
      try {
        ({ job, body } = taskPayload(request));
      } catch {
        throw new UiError("invalidWorkspaceRequest");
      }
      break;
    }
    case "todo":
    case "idea":
      job = request.type;
      try {
        body = notebookPayload(request);
      } catch {
        throw new UiError("invalidWorkspaceRequest");
      }
      break;
    case "outbox":
    case "inbox": {
      const offset = request.offset ?? 0;
      const limit = request.limit ?? 20;
      const states =
        request.type === "inbox"
          ? ["all", "pending", "processing", "processed"]
          : ["all", "pending", "processed"];
      if (
        typeof request.agentName !== "string" ||
        !request.agentName ||
        !states.includes(request.state ?? "all") ||
        !Number.isSafeInteger(offset) ||
        offset < 0 ||
        !Number.isSafeInteger(limit) ||
        limit < 1 ||
        limit > 100
      )
        throw new UiError("invalidWorkspaceRequest");
      job =
        request.type === "inbox" ? "list_agent_inbox" : "list_task_reminders";
      body = {
        agent_name: request.agentName,
        state: request.state ?? "all",
        offset,
        limit,
      };
      break;
    }
    case "reminder":
      if (
        typeof request.agentName !== "string" ||
        !request.agentName ||
        typeof request.id !== "string" ||
        !request.id ||
        request.id.length > 8192 ||
        request.id.includes("\0")
      )
        throw new UiError("invalidWorkspaceRequest");
      job = "read_agent_reminder";
      body = {
        agent_name: request.agentName,
        message_id: request.id,
      };
      break;
    case "list":
      job = "list_workspace";
      body = {
        path: relativePath(request.path),
        recursive: true,
        sort_by: "mtime",
        extensions: [
          "md",
          "mdx",
          "txt",
          "csv",
          "json",
          "jsonl",
          "yaml",
          "yml",
          "png",
          "jpg",
          "jpeg",
          "webp",
          "gif",
          "bmp",
        ],
        limit: 5000,
      };
      break;
    case "image":
      job = "read_image";
      body = { path: relativePath(request.path) };
      break;
    case "load":
      job = "load_file";
      body = { path: relativePath(request.path) };
      break;
    case "tags":
      job = "list_tags";
      body = { page, page_size: 60, order_by: "file_count" };
      break;
    case "tagFiles":
      job = "list_tag_files";
      body = {
        tags: request.tags,
        path: relativePath(request.path),
        page,
        page_size: 100,
      };
      break;
    case "search":
      if (
        typeof request.query !== "string" ||
        !request.query.trim() ||
        request.query.length > 10000
      )
        throw new UiError("invalidWorkspaceRequest");
      job = "search_workspace";
      body = {
        query: request.query,
        tags: request.tags,
        limit: 50,
        search_filter: request.path
          ? { path_prefix: relativePath(request.path) + "/" }
          : {},
        expand_links: false,
      };
      break;
    default:
      throw new UiError("invalidWorkspaceRequest");
  }
  return client.request(job, body, signal, { maxBytes: 12 * 1024 * 1024 });
}
