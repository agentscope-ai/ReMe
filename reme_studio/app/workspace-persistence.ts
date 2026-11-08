import type { ChatBlock, ChatMessage, WorkspaceTab } from "./types";

export interface PersistedWorkspaceState {
  tabs: WorkspaceTab[];
  activeTabId?: string;
}

function finishInterruptedMessage(message: ChatMessage): ChatMessage {
  const blocks = (message.blocks || []).map((block): ChatBlock => {
    if (block.type === "content" || block.type === "error") return block;
    return {
      ...block,
      status: block.status === "done" ? "done" : "error",
      expanded: false,
    };
  });
  return { ...message, blocks };
}

/** Keep reopenable UI state without making cached files the workspace source of truth. */
export function prepareWorkspaceSnapshot(
  tabs: WorkspaceTab[],
  activeTabId?: string,
): PersistedWorkspaceState {
  tabs = tabs.map(
    (tab): WorkspaceTab =>
      tab.type === "agent" && tab.mode === undefined
        ? { ...tab, mode: "chat" }
        : tab,
  );
  const isPa = (tab: WorkspaceTab) => tab.type === "agent" && tab.mode === "pa";
  const agent =
    tabs.find((tab) => isPa(tab) && tab.id === activeTabId) ?? tabs.find(isPa);
  const persistedTabs = tabs
    .filter(
      (tab) =>
        tab.type !== "agent" ||
        tab.mode === "chat" ||
        (isPa(tab) && tab === agent),
    )
    .map((tab): WorkspaceTab => {
      if (tab.type === "agent") {
        return {
          ...tab,
          streaming: false,
          messages: tab.messages.map((message) =>
            message.role === "assistant"
              ? finishInterruptedMessage(message)
              : message,
          ),
        };
      }

      if (tab.type === "graph") return tab;

      const dirty = tab.content !== tab.savedContent;
      if (dirty) return { ...tab, loading: false, error: undefined };
      return {
        ...tab,
        content: "",
        savedContent: "",
        loading: true,
        error: undefined,
      };
    });
  return {
    tabs: persistedTabs,
    activeTabId: persistedTabs.some((tab) => tab.id === activeTabId)
      ? activeTabId
      : persistedTabs.at(-1)?.id,
  };
}
