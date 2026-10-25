"use client";

import { create } from "zustand";
import { IS_DEMO } from "./studio-mode";
import { useLanguageStore } from "./i18n";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  applyStreamChunk,
  finishChatMessage,
  toggleChatBlock,
} from "./chat-stream";
import type {
  ChatMessage,
  MemoryGraphRoot,
  StreamChunk,
  WorkspaceTab,
} from "./types";
import {
  prepareWorkspaceSnapshot,
  type PersistedWorkspaceState,
} from "./workspace-persistence";
import { markMarkdownContentSaved } from "./markdown-save";
import { unsavedTabsClosedBy } from "./tab-close";

interface WorkspaceState {
  tabs: WorkspaceTab[];
  activeTabId?: string;
  openMarkdown: (path: string) => string;
  openAgent: () => string;
  openGraph: (root: MemoryGraphRoot) => string;
  closeTab: (id: string, discardUnsaved?: boolean) => void;
  closeOtherTabs: (id: string, discardUnsaved?: boolean) => void;
  setActiveTab: (id: string) => void;
  hydrateMarkdown: (id: string, content: string, mtime?: string) => void;
  failMarkdown: (id: string, error: string) => void;
  updateMarkdown: (id: string, content: string) => void;
  markSaved: (id: string, content: string, mtime?: string) => void;
  addChatTurn: (
    tabId: string,
    user: ChatMessage,
    assistant: ChatMessage,
  ) => void;
  applyChatChunk: (
    tabId: string,
    messageId: string,
    chunk: StreamChunk,
  ) => void;
  toggleChatBlock: (
    tabId: string,
    messageId: string,
    blockId: string,
    expanded: boolean,
  ) => void;
  finishChat: (tabId: string, messageId: string, error?: string) => void;
}

function updateChatMessage(
  tab: Extract<WorkspaceTab, { type: "agent" }>,
  messageId: string,
  update: (message: ChatMessage) => ChatMessage,
) {
  return {
    ...tab,
    messages: tab.messages.map((message) =>
      message.id === messageId ? update(message) : message,
    ),
  };
}

const fileTitle = (path: string) => path.split("/").pop() || path;
// Preserve the original key so existing tabs, chats, and unsaved drafts survive the Studio rename.
const WORKSPACE_STORAGE_KEY = IS_DEMO
  ? `reme-demo-tabs-v1-${useLanguageStore.getState().language}`
  : "reme-workspace";

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => {
      const updateTab = <Type extends WorkspaceTab["type"]>(
        id: string,
        type: Type,
        update: (tab: Extract<WorkspaceTab, { type: Type }>) => WorkspaceTab,
      ) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === id && tab.type === type
              ? update(tab as Extract<WorkspaceTab, { type: Type }>)
              : tab,
          ),
        }));
      return {
        tabs: [],
        activeTabId: undefined,
        openMarkdown: (path) => {
          const existing = get().tabs.find(
            (tab) => tab.type === "markdown" && tab.path === path,
          );
          if (existing) {
            set({ activeTabId: existing.id });
            return existing.id;
          }
          const id = `file:${path}`;
          set((state) => ({
            tabs: [
              ...state.tabs,
              {
                id,
                type: "markdown",
                title: fileTitle(path),
                path,
                content: "",
                savedContent: "",
                loading: true,
              },
            ],
            activeTabId: id,
          }));
          return id;
        },
        openAgent: () => {
          const id = `agent:${crypto.randomUUID()}`;
          set((state) => ({
            tabs: [
              ...state.tabs,
              { id, type: "agent", title: "", messages: [] },
            ],
            activeTabId: id,
          }));
          return id;
        },
        openGraph: (root) => {
          const id = `graph:${root}`;
          const existing = get().tabs.find(
            (tab) => tab.id === id && tab.type === "graph",
          );
          if (existing) {
            set({ activeTabId: id });
            return id;
          }
          set((state) => ({
            tabs: [
              ...state.tabs,
              { id, type: "graph", title: `${root} graph`, root },
            ],
            activeTabId: id,
          }));
          return id;
        },
        closeTab: (id, discardUnsaved = false) =>
          set((state) => {
            if (
              !discardUnsaved &&
              unsavedTabsClosedBy(state.tabs, id, false).length
            )
              return state;
            const index = state.tabs.findIndex((tab) => tab.id === id);
            const tabs = state.tabs.filter((tab) => tab.id !== id);
            const activeTabId =
              state.activeTabId === id
                ? tabs[Math.max(0, index - 1)]?.id
                : state.activeTabId;
            return { tabs, activeTabId };
          }),
        closeOtherTabs: (id, discardUnsaved = false) =>
          set((state) => {
            if (
              !discardUnsaved &&
              unsavedTabsClosedBy(state.tabs, id, true).length
            )
              return state;
            const tab = state.tabs.find((item) => item.id === id);
            return tab ? { tabs: [tab], activeTabId: id } : state;
          }),
        setActiveTab: (activeTabId) => set({ activeTabId }),
        hydrateMarkdown: (id, content, mtime) =>
          updateTab(id, "markdown", (tab) => ({
            ...tab,
            content,
            savedContent: content,
            mtime,
            loading: false,
            error: undefined,
          })),
        failMarkdown: (id, error) =>
          updateTab(id, "markdown", (tab) => ({
            ...tab,
            loading: false,
            error,
          })),
        updateMarkdown: (id, content) =>
          updateTab(id, "markdown", (tab) => ({ ...tab, content })),
        markSaved: (id, content, mtime) =>
          updateTab(id, "markdown", (tab) =>
            markMarkdownContentSaved(tab, content, mtime),
          ),
        addChatTurn: (tabId, user, assistant) =>
          updateTab(tabId, "agent", (tab) => ({
            ...tab,
            title: tab.messages.length ? tab.title : user.content.slice(0, 18),
            messages: [...tab.messages, user, assistant],
            streaming: true,
          })),
        applyChatChunk: (tabId, messageId, chunk) =>
          updateTab(tabId, "agent", (tab) => ({
            ...updateChatMessage(tab, messageId, (message) =>
              applyStreamChunk(message, chunk),
            ),
            sessionId: chunk.session_id || tab.sessionId,
          })),
        toggleChatBlock: (tabId, messageId, blockId, expanded) =>
          updateTab(tabId, "agent", (tab) =>
            updateChatMessage(tab, messageId, (message) =>
              toggleChatBlock(message, blockId, expanded),
            ),
          ),
        finishChat: (tabId, messageId, error) =>
          updateTab(tabId, "agent", (tab) => ({
            ...updateChatMessage(tab, messageId, (message) =>
              finishChatMessage(message, error),
            ),
            streaming: false,
          })),
      };
    },
    {
      name: WORKSPACE_STORAGE_KEY,
      version: 1,
      storage: createJSONStorage(() => localStorage),
      skipHydration: true,
      partialize: (state): PersistedWorkspaceState =>
        prepareWorkspaceSnapshot(state.tabs, state.activeTabId),
      merge: (persisted, current) => ({
        ...current,
        ...(persisted as PersistedWorkspaceState),
      }),
    },
  ),
);
