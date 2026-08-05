"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { SparkMenuExpandLine, SparkMenuFoldLine } from "@agentscope-ai/icons";
import {
  BookOpenText,
  Bot,
  Check,
  CircleAlert,
  FileText,
  LoaderCircle,
  Monitor,
  Moon,
  Network,
  Send,
  Settings,
  Sparkles,
  Sun,
  SunMoon,
  X,
} from "lucide-react";
import { getReMeVersion, readWorkspaceFile, streamChat } from "./api";
import { formatStreamPayloads } from "./chat-stream";
import FilesNavigator from "./files-workspace/FilesNavigator";
import MemoryGraphView from "./files-workspace/MemoryGraphView";
import { clampNavigatorWidth } from "./files-workspace/panel-resize";
import { useI18n, useLanguageStore } from "./i18n";
import { useWorkspaceStore } from "./store";
import { type ThemePreference, useThemeStore } from "./theme";
import type {
  ChatBlock,
  ChatMessage,
  DetailBlock,
  WorkspaceTab,
} from "./types";
import {
  appendWorkspaceFileReference,
  WORKSPACE_FILE_DRAG_TYPE,
} from "./workspace-drag";
import SettingsCenter from "./settings-center";

const TabbedEditor = dynamic(() => import("./files-workspace/TabbedEditor"), {
  ssr: false,
});

function GitHubIcon({ size = 18 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.23c-3.23.7-3.91-1.37-3.91-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.78 1.2 1.78 1.2 1.04 1.77 2.72 1.26 3.38.96.1-.75.4-1.26.74-1.55-2.58-.29-5.29-1.29-5.29-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.47.11-3.05 0 0 .97-.31 3.16 1.18A11 11 0 0 1 12 6.11c.98 0 1.96.13 2.88.39 2.2-1.49 3.16-1.18 3.16-1.18.63 1.58.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.4-2.72 5.38-5.3 5.67.42.36.79 1.07.79 2.16v3.25c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z" />
    </svg>
  );
}

function ThemeMenu() {
  const { t } = useI18n();
  const menuRef = useRef<HTMLDetailsElement>(null);
  const preference = useThemeStore((state) => state.preference);
  const setPreference = useThemeStore((state) => state.setPreference);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node))
        menuRef.current?.removeAttribute("open");
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  const options: Array<[ThemePreference, string, React.ReactNode]> = [
    ["light", t("lightTheme"), <Sun key="light" size={15} />],
    ["dark", t("darkTheme"), <Moon key="dark" size={15} />],
    ["system", t("systemTheme"), <Monitor key="system" size={15} />],
  ];
  return (
    <details className="theme-menu" ref={menuRef}>
      <summary aria-label={t("appearance")} title={t("appearance")}>
        <SunMoon size={18} />
      </summary>
      <div role="menu">
        {options.map(([value, label, icon]) => (
          <button
            key={value}
            role="menuitemradio"
            aria-checked={preference === value}
            className={preference === value ? "active" : ""}
            onClick={() => {
              setPreference(value);
              menuRef.current?.removeAttribute("open");
            }}
          >
            {icon}
            <span>{label}</span>
            {preference === value && <Check size={14} />}
          </button>
        ))}
      </div>
    </details>
  );
}

function Tabs() {
  const { t } = useI18n();
  const { tabs, activeTabId, setActiveTab, closeTab, closeOtherTabs } =
    useWorkspaceStore();
  const [contextMenu, setContextMenu] = useState<{
    tabId: string;
    left: number;
    top: number;
  }>();
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!contextMenu) return;
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node))
        setContextMenu(undefined);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setContextMenu(undefined);
    };
    const closeOnViewportChange = () => setContextMenu(undefined);
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", closeOnEscape);
    window.addEventListener("resize", closeOnViewportChange);
    window.addEventListener("scroll", closeOnViewportChange, true);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", closeOnEscape);
      window.removeEventListener("resize", closeOnViewportChange);
      window.removeEventListener("scroll", closeOnViewportChange, true);
    };
  }, [contextMenu]);
  return (
    <>
      <div className="tabs" role="tablist">
        {tabs.map((tab) => {
          const dirty =
            tab.type === "markdown" && tab.content !== tab.savedContent;
          return (
            <button
              key={tab.id}
              className={`tab ${tab.id === activeTabId ? "active" : ""}`}
              onClick={() => setActiveTab(tab.id)}
              onContextMenu={(event) => {
                event.preventDefault();
                setContextMenu({
                  tabId: tab.id,
                  left: Math.max(
                    8,
                    Math.min(event.clientX, window.innerWidth - 176),
                  ),
                  top: Math.max(
                    8,
                    Math.min(event.clientY, window.innerHeight - 82),
                  ),
                });
              }}
              role="tab"
            >
              {tab.type === "agent" ? (
                <Bot size={14} />
              ) : tab.type === "graph" ? (
                <Network size={14} />
              ) : (
                <FileText size={14} />
              )}
              <span>
                {tab.type === "markdown"
                  ? tab.path.split("/").slice(-2).join("/")
                  : tab.type === "graph"
                  ? `${tab.root} · ${t("memoryGraphShort")}`
                  : tab.title || t("newConversation")}
              </span>
              {dirty ? (
                <i className="dirty" />
              ) : (
                <X
                  size={13}
                  onClick={(event) => {
                    event.stopPropagation();
                    closeTab(tab.id);
                  }}
                />
              )}
            </button>
          );
        })}
      </div>
      {contextMenu && (
        <div
          className="tab-context-menu"
          ref={menuRef}
          role="menu"
          style={{ left: contextMenu.left, top: contextMenu.top }}
        >
          <button
            role="menuitem"
            onClick={() => {
              closeTab(contextMenu.tabId);
              setContextMenu(undefined);
            }}
          >
            {t("closeCurrentTab")}
          </button>
          <button
            role="menuitem"
            disabled={tabs.length <= 1}
            onClick={() => {
              closeOtherTabs(contextMenu.tabId);
              setContextMenu(undefined);
            }}
          >
            {t("closeOtherTabs")}
          </button>
        </div>
      )}
    </>
  );
}

function MarkdownView({ content }: { content: string }) {
  return (
    <article className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </article>
  );
}

function ChunkStatus({
  active,
  error = false,
}: {
  active: boolean;
  error?: boolean;
}) {
  if (error) return <CircleAlert size={13} />;
  return active ? (
    <LoaderCircle className="spin" size={13} />
  ) : (
    <Check size={13} />
  );
}

function detailLabel(
  block: DetailBlock,
  t: ReturnType<typeof useI18n>["t"],
): string {
  if (block.type === "think") return t("thinking");
  if (block.type === "data")
    return block.mediaType
      ? `${t("dataChunk")} · ${block.mediaType}`
      : t("dataChunk");
  if (block.type === "approval") return t("approvalChunk");
  if (block.type === "usage") return t("usageChunk");
  return t("unknownChunk", { type: block.sourceType });
}

function ChatBlockView({
  block,
  onToggle,
}: {
  block: ChatBlock;
  onToggle: (blockId: string, expanded: boolean) => void;
}) {
  const { t } = useI18n();
  if (block.type === "content") return <MarkdownView content={block.text} />;
  if (block.type === "error")
    return (
      <div className="chunk-error">
        <CircleAlert size={14} />
        {block.text}
      </div>
    );
  if (block.type === "usage") return null;

  const active =
    block.status === "streaming" ||
    block.status === "calling" ||
    block.status === "running";
  const failed = block.status === "error";
  if (block.type === "tool") {
    const call = formatStreamPayloads(block.callPayloads);
    const result = formatStreamPayloads(block.resultPayloads);
    return (
      <details
        className={`stream-block tool ${failed ? "failed" : ""}`}
        open={block.expanded}
        onToggle={(event) => {
          if (event.currentTarget.open !== block.expanded)
            onToggle(block.id, event.currentTarget.open);
        }}
      >
        <summary>
          <ChunkStatus active={active} error={failed} />
          <span>{block.name}</span>
          <small>{active ? t("streaming") : t("completed")}</small>
        </summary>
        <div className="stream-block-body">
          {call && (
            <section>
              <strong>{t("toolCall")}</strong>
              <pre>{call}</pre>
            </section>
          )}
          {result && (
            <section>
              <strong>{t("toolResult")}</strong>
              <pre className="tool-result-scroll">{result}</pre>
            </section>
          )}
        </div>
      </details>
    );
  }

  const detail = formatStreamPayloads(block.payloads);
  return (
    <details
      className={`stream-block ${block.type} ${failed ? "failed" : ""}`}
      open={block.expanded}
      onToggle={(event) => {
        if (event.currentTarget.open !== block.expanded)
          onToggle(block.id, event.currentTarget.open);
      }}
    >
      <summary>
        <ChunkStatus active={active} error={failed} />
        <span>{detailLabel(block, t)}</span>
        <small>{active ? t("streaming") : t("completed")}</small>
      </summary>
      {detail && (
        <div className="stream-block-body">
          <pre>{detail}</pre>
        </div>
      )}
    </details>
  );
}

function Chat({ tab }: { tab: Extract<WorkspaceTab, { type: "agent" }> }) {
  const { t } = useI18n();
  const [input, setInput] = useState("");
  const [fileDragOver, setFileDragOver] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const controller = useRef<AbortController | null>(null);
  const { addChatTurn, applyChatChunk, toggleChatBlock, finishChat } =
    useWorkspaceStore();

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [tab.messages]);
  useEffect(() => () => controller.current?.abort(), []);

  const send = async () => {
    const query = input.trim();
    if (!query || tab.streaming) return;
    setInput("");
    const user: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    const assistant: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      blocks: [],
    };
    addChatTurn(tab.id, user, assistant);
    controller.current = new AbortController();
    try {
      await streamChat(
        query,
        tab.sessionId,
        controller.current.signal,
        (chunk) => applyChatChunk(tab.id, assistant.id, chunk),
      );
      finishChat(tab.id, assistant.id);
    } catch (error) {
      if (!controller.current.signal.aborted)
        finishChat(
          tab.id,
          assistant.id,
          error instanceof Error ? error.message : t("chatFailed"),
        );
    }
  };

  return (
    <div className="chat">
      <div className="messages">
        {!tab.messages.length && (
          <div className="chat-empty">
            <div className="agent-logo">
              <Sparkles size={24} />
            </div>
            <h1>{t("chatTitle")}</h1>
            <p>{t("chatDescription")}</p>
            <div className="suggestions">
              {[t("promptRecent"), t("promptTasks"), t("promptIdeas")].map(
                (text) => (
                  <button key={text} onClick={() => setInput(text)}>
                    {text}
                  </button>
                ),
              )}
            </div>
          </div>
        )}
        {tab.messages.map((message) => (
          <div className={`message ${message.role}`} key={message.id}>
            <div className="avatar">
              {message.role === "user" ? (
                t("you")
              ) : (
                <span aria-label="ReMe" title="ReMe">
                  R
                </span>
              )}
            </div>
            <div className="bubble">
              {message.role === "user" && <p>{message.content}</p>}
              {message.role === "assistant" &&
                message.blocks?.map((block) => (
                  <ChatBlockView
                    key={block.id}
                    block={block}
                    onToggle={(blockId, expanded) =>
                      toggleChatBlock(tab.id, message.id, blockId, expanded)
                    }
                  />
                ))}
              {message.role === "assistant" &&
                tab.streaming &&
                !message.blocks?.some((block) => block.type !== "usage") && (
                  <LoaderCircle className="spin" size={16} />
                )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div
        className={`composer ${fileDragOver ? "file-drag-over" : ""}`}
        onDragOver={(event) => {
          if (!event.dataTransfer.types.includes(WORKSPACE_FILE_DRAG_TYPE))
            return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
          setFileDragOver(true);
        }}
        onDragLeave={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node | null))
            setFileDragOver(false);
        }}
        onDrop={(event) => {
          const path = event.dataTransfer.getData(WORKSPACE_FILE_DRAG_TYPE);
          if (!path) return;
          event.preventDefault();
          setFileDragOver(false);
          setInput((current) => appendWorkspaceFileReference(current, path));
        }}
      >
        <div>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder={t("askWorkspace")}
            rows={1}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
          />
          <button
            onClick={send}
            disabled={!input.trim() || tab.streaming}
            aria-label={t("send")}
          >
            <Send size={17} />
          </button>
        </div>
        <span>{t("composerHint")}</span>
      </div>
    </div>
  );
}

function Workspace() {
  const { language, setLanguage, t } = useI18n();
  const hydrateLanguage = useLanguageStore((state) => state.hydrate);
  const hydrateTheme = useThemeStore((state) => state.hydrate);
  const [navOpen, setNavOpen] = useState(true);
  const [navigatorWidth, setNavigatorWidth] = useState(260);
  const [resizingNavigator, setResizingNavigator] = useState(false);
  const [version, setVersion] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { tabs, activeTabId, openAgent, hydrateMarkdown, failMarkdown } =
    useWorkspaceStore();
  const active = tabs.find((tab) => tab.id === activeTabId);
  useEffect(() => {
    let mounted = true;
    void Promise.resolve(useWorkspaceStore.persist.rehydrate()).then(
      async () => {
        const restoredFiles = useWorkspaceStore
          .getState()
          .tabs.filter(
            (tab): tab is Extract<WorkspaceTab, { type: "markdown" }> =>
              tab.type === "markdown" && Boolean(tab.loading),
          );
        await Promise.all(
          restoredFiles.map(async (tab) => {
            try {
              const file = await readWorkspaceFile(tab.path);
              if (mounted)
                hydrateMarkdown(tab.id, file.content, file.stat.mtime);
            } catch (error) {
              if (mounted)
                failMarkdown(
                  tab.id,
                  error instanceof Error
                    ? error.message
                    : "Failed to read file",
                );
            }
          }),
        );
      },
    );
    return () => {
      mounted = false;
    };
  }, [failMarkdown, hydrateMarkdown]);
  useEffect(() => {
    hydrateLanguage();
  }, [hydrateLanguage]);
  useEffect(() => {
    hydrateTheme();
  }, [hydrateTheme]);
  useEffect(() => {
    const controller = new AbortController();
    void getReMeVersion()
      .then((nextVersion) => {
        if (!controller.signal.aborted) setVersion(nextVersion);
      })
      .catch(() => undefined);
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const saved = Number(localStorage.getItem("reme-navigator-width"));
    if (!Number.isFinite(saved) || saved <= 0) return;
    const frame = requestAnimationFrame(() =>
      setNavigatorWidth(clampNavigatorWidth(saved, window.innerWidth)),
    );
    return () => cancelAnimationFrame(frame);
  }, []);

  const resizeNavigator = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const navigator = event.currentTarget.parentElement;
    const surface = navigator?.parentElement;
    if (!navigator || !surface) return;
    setResizingNavigator(true);
    const startX = event.clientX;
    const initialWidth = navigator.getBoundingClientRect().width;
    const containerWidth = surface.getBoundingClientRect().width;
    const move = (nextEvent: PointerEvent) => {
      setNavigatorWidth(
        clampNavigatorWidth(
          initialWidth + nextEvent.clientX - startX,
          containerWidth,
        ),
      );
    };
    const stop = (nextEvent: PointerEvent) => {
      const finalWidth = clampNavigatorWidth(
        initialWidth + nextEvent.clientX - startX,
        containerWidth,
      );
      setNavigatorWidth(finalWidth);
      localStorage.setItem("reme-navigator-width", String(finalWidth));
      setResizingNavigator(false);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  };

  const resizeNavigatorWithKeyboard = (
    event: React.KeyboardEvent<HTMLDivElement>,
  ) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const containerWidth =
      event.currentTarget.parentElement?.parentElement?.getBoundingClientRect()
        .width ?? window.innerWidth;
    setNavigatorWidth((current) => {
      const next = clampNavigatorWidth(
        current + (event.key === "ArrowRight" ? 24 : -24),
        containerWidth,
      );
      localStorage.setItem("reme-navigator-width", String(next));
      return next;
    });
  };

  return (
    <main
      className={`shell ${navOpen ? "" : "nav-closed"} ${
        resizingNavigator ? "navigator-is-resizing" : ""
      }`}
      style={
        { "--navigator-width": `${navigatorWidth}px` } as React.CSSProperties
      }
    >
      <header className="topbar">
        <button
          className={`menu ${navOpen ? "is-open" : ""}`}
          onClick={() => setNavOpen(!navOpen)}
          aria-label={t("toggleNavigator")}
          aria-expanded={navOpen}
          aria-controls="workspace-navigator"
        >
          <span className="menu-icons" aria-hidden="true">
            <SparkMenuFoldLine size={20} className="menu-fold-icon" />
            <SparkMenuExpandLine size={20} className="menu-expand-icon" />
          </span>
        </button>
        <strong>
          ReMe
          {version && (
            <>
              <span className="app-version-divider" aria-hidden="true" />
              <span className="app-version">v{version}</span>
            </>
          )}
        </strong>
        <span>
          {active?.type === "markdown"
            ? active.path
            : active?.type === "graph"
            ? `${active.root} · ${t("memoryGraph")}`
            : active?.title || t("workspace")}
        </span>
        <div className="topbar-actions">
          <nav className="resource-links" aria-label={t("documentation")}>
            <a
              href="https://docs.agentscope.io/reme/latest/en/overview"
              target="_blank"
              rel="noreferrer"
            >
              <BookOpenText size={17} />
              <span>{t("documentation")}</span>
            </a>
            <i aria-hidden="true" />
            <a
              href="https://github.com/modelscope/ReMe"
              target="_blank"
              rel="noreferrer"
            >
              <GitHubIcon />
              <span>{t("github")}</span>
            </a>
          </nav>
          <div className="topbar-divider" aria-hidden="true" />
          <div
            className="language-switch"
            aria-label={t("switchLanguage")}
            role="group"
          >
            <button
              className={language === "zh" ? "active" : ""}
              onClick={() => setLanguage("zh")}
            >
              中文
            </button>
            <button
              className={language === "en" ? "active" : ""}
              onClick={() => setLanguage("en")}
            >
              EN
            </button>
          </div>
          <ThemeMenu />
          <button
            className="settings-trigger"
            onClick={() => setSettingsOpen(true)}
            aria-label={t("settings")}
            title={t("settings")}
          >
            <Settings size={18} />
          </button>
        </div>
      </header>
      <div className="surface">
        <FilesNavigator
          open={navOpen}
          width={navigatorWidth}
          resizing={resizingNavigator}
          onResizeStart={resizeNavigator}
          onResizeKeyDown={resizeNavigatorWithKeyboard}
        />
        <section className="workbench">
          <Tabs />
          <div className="content">
            {!active && (
              <div className="welcome">
                <div className="agent-logo">R</div>
                <h1>ReMe Workspace</h1>
                <p>{t("welcomeDescription")}</p>
                <button onClick={openAgent}>
                  <Sparkles size={16} />
                  {t("startChat")}
                </button>
                <small>{t("localFiles")}</small>
              </div>
            )}
            {active?.type === "markdown" && (
              <TabbedEditor key={active.id} tab={active} />
            )}
            {active?.type === "agent" && <Chat tab={active} />}
            {active?.type === "graph" && (
              <MemoryGraphView key={active.id} root={active.root} />
            )}
          </div>
        </section>
      </div>
      <SettingsCenter
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </main>
  );
}

export function ReMeWorkspace() {
  return <Workspace />;
}
