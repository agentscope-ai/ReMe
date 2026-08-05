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
      status: block.status === "error" ? "error" : "done",
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
  const persistedTabs = tabs.map((tab): WorkspaceTab => {
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
