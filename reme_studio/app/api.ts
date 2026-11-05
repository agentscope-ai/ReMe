import type {
  AppConfig,
  FileStat,
  GraphSnapshot,
  ReMeHealth,
  ReMeResponse,
  StreamChunk,
  SessionSnapshot,
} from "./types";
import { readChatStream } from "./chat-stream";
import { healthFromResponse } from "./health-status";
import { translate, useLanguageStore, type TranslationKey } from "./i18n";
import {
  WORKSPACE_FILE_LIMIT,
  workspaceFileListing,
  type WorkspaceFileListing,
} from "./workspace-files";
import { displayReMeApiEndpoint, normalizeReMeApiUrl } from "./api-endpoint";

export const REME_API_URL = normalizeReMeApiUrl(
  process.env.NEXT_PUBLIC_REME_API_URL || "http://127.0.0.1:2333",
);
export const REME_API_ENDPOINT = displayReMeApiEndpoint(
  REME_API_URL,
  typeof window === "undefined" ? undefined : window.location.origin,
);
const message = (key: TranslationKey, status: number) =>
  translate(useLanguageStore.getState().language, key, {
    status: String(status),
  });

async function parseResponse<T>(response: Response): Promise<ReMeResponse<T>> {
  let payload: ReMeResponse<T>;
  try {
    payload = (await response.json()) as ReMeResponse<T>;
  } catch {
    throw new Error(message("invalidResponse", response.status));
  }
  if (!response.ok || !payload.success) {
    const detail =
      typeof payload.answer === "string"
        ? payload.answer
        : message("requestFailed", response.status);
    throw new Error(detail);
  }
  return payload;
}

export async function callReMe<T>(
  action: string,
  body: Record<string, unknown> = {},
): Promise<ReMeResponse<T>> {
  const response = await fetch(`${REME_API_URL}/${action}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseResponse<T>(response);
}

export async function getAppConfig(): Promise<AppConfig> {
  return (await callReMe<AppConfig>("app_config")).answer;
}

export async function getReMeVersion(): Promise<string> {
  return String((await callReMe<string>("version")).answer);
}

export async function readChatSession(
  sessionId: string,
  signal: AbortSignal,
): Promise<SessionSnapshot | undefined> {
  const response = await fetch(`${REME_API_URL}/session_read`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
    signal,
  });
  if (response.status === 404) return undefined;
  const payload = (await response.json()) as ReMeResponse<SessionSnapshot>;
  if (payload.metadata?.code === "session_not_supported") return undefined;
  if (!response.ok || !payload.success) {
    throw new Error(
      typeof payload.answer === "string"
        ? payload.answer
        : message("requestFailed", response.status),
    );
  }
  return payload.answer;
}

export async function getReMeStatus(): Promise<ReMeResponse<string>> {
  return callReMe<string>("status");
}

export async function getReMeHealth(): Promise<ReMeHealth | undefined> {
  const response = await callReMe<string>("health_check");
  return healthFromResponse(response);
}

export async function rebuildReMeIndex(): Promise<ReMeResponse<unknown>> {
  return callReMe<unknown>("reindex");
}

export async function getGraphSnapshot(): Promise<GraphSnapshot> {
  return (await callReMe<GraphSnapshot>("graph_snapshot")).answer;
}

export async function listWorkspaceFiles(
  extensions: string[],
): Promise<WorkspaceFileListing> {
  const response = await callReMe<string>("list", {
    path: "",
    recursive: true,
    limit: WORKSPACE_FILE_LIMIT,
    sort_by: "mtime",
    extensions,
  });
  return workspaceFileListing(response.metadata.items);
}

export async function readWorkspaceFile(
  path: string,
): Promise<{ content: string; stat: FileStat }> {
  const response = await callReMe<string>("load", { path });
  return {
    content: String(response.answer ?? ""),
    stat: response.metadata as unknown as FileStat,
  };
}

export async function saveWorkspaceFile(
  path: string,
  content: string,
  expectedMtime?: string,
): Promise<FileStat> {
  const response = await callReMe<string>("save", {
    path,
    content,
    expected_mtime: expectedMtime || null,
  });
  return response.metadata as unknown as FileStat;
}

export async function streamChat(
  query: string,
  sessionId: string | undefined,
  signal: AbortSignal,
  onChunk: (chunk: StreamChunk) => void,
): Promise<void> {
  const response = await fetch(`${REME_API_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ query, session_id: sessionId || null }),
    signal,
  });
  if (!response.ok || !response.body)
    throw new Error(message("agentUnavailable", response.status));
  if (
    response.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !==
    "text/event-stream"
  ) {
    await response.body.cancel();
    throw new Error(message("invalidResponse", response.status));
  }
  await readChatStream(
    response.body,
    onChunk,
    translate(useLanguageStore.getState().language, "chatInterrupted"),
  );
}
