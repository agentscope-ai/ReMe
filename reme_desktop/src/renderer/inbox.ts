import type { JobResponse, WorkspaceRequest } from "../shared/protocol.ts";
import { deserializeError, translator, type Language } from "../shared/i18n.ts";

export async function inboxRequest<T>(value: WorkspaceRequest): Promise<T> {
  const response = await window.desktop.workspace(value);
  if ("error" in response) throw deserializeError(response.error);
  if (!response.success) throw new Error(String(response.answer));
  return (response as JobResponse<T>).answer;
}

export function inboxDate(
  timestamp: string | null | undefined,
  language: Language,
): string {
  const date = timestamp ? new Date(timestamp) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleString(language === "zh" ? "zh-CN" : "en", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : translator(language)("earlier");
}
