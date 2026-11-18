import { useEffect, useState } from "react";
import { fromMarkdown } from "mdast-util-from-markdown";
import type { ResourcePage, WorkspaceRequest } from "../shared/protocol.ts";
import { deserializeError, translator, type Language } from "../shared/i18n.ts";

export function resourceSource(source: string, language: Language): string {
  const key =
    source === "codex_dialogue"
      ? "sourceConversations"
      : source === "screenshot"
        ? "sourceScreenshots"
        : null;
  return key ? translator(language)(key) : source;
}

export async function requestResource<T>(value: WorkspaceRequest): Promise<T> {
  const response = await window.desktop.workspace(value);
  if ("error" in response) throw deserializeError(response.error);
  if (!response.success) throw new Error(String(response.answer));
  return response.answer as T;
}

export function formatDate(
  value: string | null | undefined,
  language: Language,
): string {
  const date = value ? new Date(value) : null;
  return date && Number.isFinite(date.getTime())
    ? date.toLocaleString(language === "zh" ? "zh-CN" : "en", {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "—";
}

export function plainText(content: string): string {
  const text = (node: {
    type?: string;
    value?: string;
    alt?: string;
    children?: unknown[];
  }): string =>
    node.value ??
    node.alt ??
    node.children
      ?.map((child) => text(child as Parameters<typeof text>[0]))
      .join(
        ["root", "list", "listItem", "blockquote"].includes(node.type ?? "")
          ? " "
          : "",
      ) ??
    "";
  return text(fromMarkdown(content)).replace(/\s+/g, " ").trim();
}

/** One visible resource page owns one polling loop; details reuse its canonical rows. */
export function useResourcePage<T>(
  request: WorkspaceRequest,
  scope: string,
  active: boolean,
  revision: number,
) {
  const key = JSON.stringify(request);
  const [page, setPage] = useState<ResourcePage<T> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    setPage(null);
    setError(null);
    setLoading(true);
  }, [key, scope]);
  useEffect(() => {
    if (!active) return;
    let live = true;
    let pending = false;
    const refresh = async () => {
      if (pending || document.hidden) return;
      pending = true;
      try {
        const value = await requestResource<ResourcePage<T>>(
          JSON.parse(key) as WorkspaceRequest,
        );
        if (live) {
          setPage(value);
          setError(null);
        }
      } catch (reason) {
        if (live) setError(reason);
      } finally {
        pending = false;
        if (live) setLoading(false);
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [key, scope, active, revision, retry]);
  return {
    page,
    error,
    loading,
    refresh: () => setRetry((value) => value + 1),
  };
}
