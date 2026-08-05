"use client";

import { create } from "zustand";
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

interface WorkspaceState {
  tabs: WorkspaceTab[];
  activeTabId?: string;
  openMarkdown: (path: string) => string;
  openAgent: () => string;
  openGraph: (root: MemoryGraphRoot) => string;
  closeTab: (id: string) => void;
  closeOtherTabs: (id: string) => void;
  setActiveTab: (id: string) => void;
  hydrateMarkdown: (id: string, content: string, mtime?: string) => void;
  failMarkdown: (id: string, error: string) => void;
  updateMarkdown: (id: string, content: string) => void;
  markSaved: (id: string, mtime?: string) => void;
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

const fileTitle = (path: string) => path.split("/").pop() || path;

export const useWorkspaceStore = create<WorkspaceState>()(
  persist(
    (set, get) => ({
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
          tabs: [...state.tabs, { id, type: "agent", title: "", messages: [] }],
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
      closeTab: (id) =>
        set((state) => {
          const index = state.tabs.findIndex((tab) => tab.id === id);
          const tabs = state.tabs.filter((tab) => tab.id !== id);
          const activeTabId =
            state.activeTabId === id
              ? tabs[Math.max(0, index - 1)]?.id
              : state.activeTabId;
          return { tabs, activeTabId };
        }),
      closeOtherTabs: (id) =>
        set((state) => {
          const tab = state.tabs.find((item) => item.id === id);
          return tab ? { tabs: [tab], activeTabId: id } : state;
        }),
      setActiveTab: (activeTabId) => set({ activeTabId }),
      hydrateMarkdown: (id, content, mtime) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === id && tab.type === "markdown"
              ? {
                  ...tab,
                  content,
                  savedContent: content,
                  mtime,
                  loading: false,
                  error: undefined,
                }
              : tab,
          ),
        })),
      failMarkdown: (id, error) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === id && tab.type === "markdown"
              ? { ...tab, loading: false, error }
              : tab,
          ),
        })),
      updateMarkdown: (id, content) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === id && tab.type === "markdown"
              ? { ...tab, content }
              : tab,
          ),
        })),
      markSaved: (id, mtime) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === id && tab.type === "markdown"
              ? { ...tab, savedContent: tab.content, mtime: mtime || tab.mtime }
              : tab,
          ),
        })),
      addChatTurn: (tabId, user, assistant) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === tabId && tab.type === "agent"
              ? {
                  ...tab,
                  title: tab.messages.length
                    ? tab.title
                    : user.content.slice(0, 18),
                  messages: [...tab.messages, user, assistant],
                  streaming: true,
                }
              : tab,
          ),
        })),
      applyChatChunk: (tabId, messageId, chunk) =>
        set((state) => ({
          tabs: state.tabs.map((tab) => {
            if (tab.id !== tabId || tab.type !== "agent") return tab;
            const messages = tab.messages.map((message) =>
              message.id === messageId
                ? applyStreamChunk(message, chunk)
                : message,
            );
            return {
              ...tab,
              sessionId: chunk.session_id || tab.sessionId,
              messages,
            };
          }),
        })),
      toggleChatBlock: (tabId, messageId, blockId, expanded) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === tabId && tab.type === "agent"
              ? {
                  ...tab,
                  messages: tab.messages.map((message) =>
                    message.id === messageId
                      ? toggleChatBlock(message, blockId, expanded)
                      : message,
                  ),
                }
              : tab,
          ),
        })),
      finishChat: (tabId, messageId, error) =>
        set((state) => ({
          tabs: state.tabs.map((tab) =>
            tab.id === tabId && tab.type === "agent"
              ? {
                  ...tab,
                  streaming: false,
                  messages: tab.messages.map((message) =>
                    message.id === messageId
                      ? finishChatMessage(message, error)
                      : message,
                  ),
                }
              : tab,
          ),
        })),
    }),
    {
      name: "reme-workspace",
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
