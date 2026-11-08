import type {
  JobResponse,
  WorkspaceConfig,
  WorkspaceRequest,
} from "../shared/protocol.ts";
import { UiError } from "../shared/i18n.ts";
import type { ReMeClient } from "./reme-client.ts";

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

/** Expose browsing jobs and explicit user-read receipts to the renderer. */
export async function requestWorkspace(
  client: ReMeClient,
  request: WorkspaceRequest,
  signal: AbortSignal,
): Promise<JobResponse> {
  if (!request || typeof request.type !== "string")
    throw new UiError("invalidWorkspaceRequest");
  if (request.type === "config") {
    const config = await client.call<{
      workspace_dir: string;
      daily_dir: string;
      digest_dir: string;
      jobs: Record<
        string,
        { enabled?: boolean; enable_serve?: boolean; backend: string }
      >;
      components?: {
        file_store?: Record<string, { tag_index?: string }>;
        tag_index?: Record<string, { tag_key?: string }>;
      };
    }>("app_config", {}, signal);
    const answer: WorkspaceConfig = {
      workspace_dir: config.workspace_dir,
      daily_dir: config.daily_dir ?? "daily",
      digest_dir: config.digest_dir ?? "digest",
      tag_key:
        config.components?.tag_index?.[
          config.components?.file_store?.default?.tag_index ?? "default"
        ]?.tag_key ?? "memory_tags",
      jobs: Object.entries(config.jobs ?? {})
        .filter(
          ([, job]) =>
            job.enabled !== false &&
            job.enable_serve !== false &&
            job.backend !== "background",
        )
        .map(([name]) => name),
    };
    return { success: true, answer, metadata: {} };
  }
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
    case "proactiveInbox": {
      const kind = request.kind ?? "all";
      const filter = request.filter ?? "all";
      const offset = request.offset ?? 0;
      const limit = request.limit ?? 5;
      if (
        !["all", "task_result", "instruction"].includes(kind) ||
        !["all", "pending", "unread"].includes(filter) ||
        !Number.isSafeInteger(offset) ||
        offset < 0 ||
        !Number.isSafeInteger(limit) ||
        limit < 1 ||
        limit > 100 ||
        (filter === "unread" && kind !== "task_result") ||
        (request.taskId !== undefined &&
          (typeof request.taskId !== "string" ||
            kind !== "task_result" ||
            !/^[A-Za-z0-9_-]{1,128}$/.test(request.taskId)))
      )
        throw new UiError("invalidWorkspaceRequest");
      job = "proactive_inbox_view";
      body = {
        action: "list",
        kind,
        filter,
        task_id: request.taskId,
        offset,
        limit,
      };
      break;
    }
    case "proactiveInboxEntry":
      if (
        typeof request.id !== "string" ||
        !request.id ||
        request.id.length > 8192 ||
        request.id.includes("\0")
      )
        throw new UiError("invalidWorkspaceRequest");
      job = "proactive_inbox_view";
      body = { action: "read", entry_id: request.id };
      break;
    case "taskResult":
      if (typeof request.markRead !== "boolean")
        throw new UiError("invalidWorkspaceRequest");
      job = "task_results";
      body = {
        action: "read",
        path: relativePath(request.path),
        mark_read: request.markRead,
      };
      break;
    case "conversation":
      if (
        typeof request.sessionId !== "string" ||
        !/^[A-Za-z0-9_-]{1,128}$/.test(request.sessionId)
      )
        throw new UiError("invalidWorkspaceRequest");
      job = "session_read";
      body = { session_id: request.sessionId };
      break;
    case "list":
      job = "list";
      body = {
        path: relativePath(request.path),
        recursive: true,
        sort_by: "mtime",
        extensions: ["md", "mdx", "txt", "csv"],
        limit: 5000,
      };
      break;
    case "load":
      job = "load";
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
      job = "search";
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
