"use client";

import { useEffect, useRef, useState } from "react";
import {
  Check,
  CircleAlert,
  LoaderCircle,
  MessageSquarePlus,
  Send,
} from "lucide-react";
import WorkspaceMarkdown from "./files-workspace/WorkspaceMarkdown";
import {
  getPaSession,
  newPaSession,
  readChatSession,
  streamChat,
  streamPaChat,
} from "./api";
import { chatStreamError, formatStreamPayloads } from "./chat-stream";
import { isChatAvailable, isJobAvailable } from "./chat-modes";
import { useI18n } from "./i18n";
import { useWorkspaceStore } from "./store";
import { IS_DEMO } from "./studio-mode";
import type {
  ChatBlock,
  AppConfig,
  ChatMessage,
  DetailBlock,
  PaSession,
  StreamChunk,
  WorkspaceTab,
} from "./types";
import {
  appendWorkspaceFileReference,
  WORKSPACE_FILE_DRAG_TYPE,
} from "./workspace-drag";
import { scenarios } from "../demo/scenarios";

function MarkdownView({ content }: { content: string }) {
  return (
    <article className="markdown">
      <WorkspaceMarkdown content={content} />
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
  const failed = block.status === "error" || block.status === "interrupted";
  const tool = block.type === "tool";
  const call = tool ? formatStreamPayloads(block.callPayloads) : "";
  const result = tool ? formatStreamPayloads(block.resultPayloads) : "";
  const detail = tool ? "" : formatStreamPayloads(block.payloads);
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
        <span>{tool ? block.name : detailLabel(block, t)}</span>
        <small>
          {t(
            block.status === "interrupted"
              ? "interrupted"
              : block.status === "waiting"
              ? "waiting"
              : failed
              ? "failed"
              : active
              ? "streaming"
              : "completed",
          )}
        </small>
      </summary>
      {tool ? (
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
      ) : detail ? (
        <div className="stream-block-body">
          <pre>{detail}</pre>
        </div>
      ) : null}
    </details>
  );
}

export default function Chat({
  tab,
  active,
  config,
}: {
  tab: Extract<WorkspaceTab, { type: "agent" }>;
  active: boolean;
  config?: AppConfig;
}) {
  const { language, t } = useI18n();
  const proactive = tab.mode === "pa";
  const available = isChatAvailable(config, tab.mode);
  const historySessionId = proactive ? undefined : tab.sessionId;
  const canReadHistory = isJobAvailable(config, "session_read", "base");
  const [input, setInput] = useState("");
  const [fileDragOver, setFileDragOver] = useState(false);
  const [syncError, setSyncError] = useState("");
  const [session, setSession] = useState<PaSession>();
  const [switching, setSwitching] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const controller = useRef<AbortController | null>(null);
  const {
    addChatTurn,
    applyChatChunk,
    toggleChatBlock,
    finishChat,
    selectPaSession,
    openAgent,
  } = useWorkspaceStore();

  useEffect(() => {
    if (active) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [tab.messages, active]);
  useEffect(() => () => controller.current?.abort(), []);

  useEffect(() => {
    if (
      IS_DEMO ||
      !available ||
      !active ||
      tab.streaming ||
      switching ||
      (!proactive && (!historySessionId || !canReadHistory))
    )
      return;
    let stopped = false;
    let historyAvailable = canReadHistory;
    let pending = false;
    let timer: ReturnType<typeof setTimeout>;
    let request: AbortController | undefined;
    const refresh = async () => {
      if (
        stopped ||
        pending ||
        (!proactive && !historyAvailable) ||
        document.visibilityState !== "visible"
      )
        return;
      pending = true;
      request = new AbortController();
      try {
        const currentSession: PaSession = proactive
          ? await getPaSession(request.signal)
          : { session_id: historySessionId ?? null, status: null };
        if (stopped || request.signal.aborted) return;
        if (proactive) {
          useWorkspaceStore
            .getState()
            .selectPaSession(tab.id, currentSession.session_id);
          setSession(currentSession);
        }
        if (currentSession.session_id && historyAvailable) {
          const current = useWorkspaceStore
            .getState()
            .tabs.find((item) => item.id === tab.id);
          const expectedLastId =
            current?.type === "agent" ? current.messages.at(-1)?.id : undefined;
          const snapshot = await readChatSession(
            currentSession.session_id,
            request.signal,
          );
          if (stopped || request.signal.aborted) return;
          if (snapshot)
            useWorkspaceStore
              .getState()
              .syncChatSession(tab.id, snapshot, expectedLastId);
          else historyAvailable = false;
        }
        setSyncError("");
      } catch (error) {
        if (!stopped && !request.signal.aborted)
          setSyncError(error instanceof Error ? error.message : String(error));
      } finally {
        pending = false;
        if (!stopped && (proactive || historyAvailable)) {
          clearTimeout(timer);
          timer = setTimeout(() => void refresh(), 5000);
        }
      }
    };
    const onVisible = () => {
      clearTimeout(timer);
      if (document.visibilityState === "visible") void refresh();
      else request?.abort();
    };
    void refresh();
    window.addEventListener("focus", onVisible);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      stopped = true;
      clearTimeout(timer);
      request?.abort();
      window.removeEventListener("focus", onVisible);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [
    active,
    available,
    proactive,
    historySessionId,
    canReadHistory,
    tab.id,
    tab.streaming,
    switching,
  ]);

  const ready = available && (!proactive || IS_DEMO || session !== undefined);
  const busy = Boolean(
    tab.streaming || switching || (proactive && session?.status === "active"),
  );
  const canSend = ready && !busy && session?.status !== "error";
  const startNewSession = async () => {
    if (!ready || busy || controller.current) return;
    if (!proactive) {
      openAgent("chat");
      return;
    }
    const request = new AbortController();
    controller.current = request;
    setSwitching(true);
    try {
      const sessionId = await newPaSession(request.signal);
      if (request.signal.aborted) return;
      selectPaSession(tab.id, sessionId);
      setSession({ session_id: sessionId, status: "idle" });
      setInput("");
      setSyncError("");
    } catch (error) {
      if (!request.signal.aborted)
        setSyncError(error instanceof Error ? error.message : String(error));
    } finally {
      if (controller.current === request) controller.current = null;
      setSwitching(false);
    }
  };

  const suggestions = (
    <div className="suggestions">
      {(IS_DEMO
        ? scenarios[language].map((scenario) => scenario.question)
        : [t("promptRecent"), t("promptTasks"), t("promptIdeas")]
      ).map((question) => (
        <button key={question} onClick={() => setInput(question)}>
          {question}
        </button>
      ))}
    </div>
  );

  const send = async () => {
    const query = input.trim();
    if (!query || !canSend || controller.current) return;
    setInput("");
    const user: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };
    const assistant: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      status: "running",
      content: "",
      blocks: [],
    };
    addChatTurn(tab.id, user, assistant);
    const requestController = new AbortController();
    controller.current = requestController;
    const error = await chatStreamError(
      () => {
        const onChunk = (chunk: StreamChunk) =>
          applyChatChunk(tab.id, assistant.id, chunk);
        return proactive
          ? streamPaChat(query, requestController.signal, onChunk)
          : streamChat(query, tab.sessionId, requestController.signal, onChunk);
      },
      requestController.signal,
      t("chatFailed"),
    );
    finishChat(tab.id, assistant.id, error);
    if (controller.current === requestController) controller.current = null;
  };

  return (
    <div className="chat" style={active ? undefined : { display: "none" }}>
      <div className="chat-session">
        <div>
          <strong>{t(proactive ? "paChat" : "chatTab")}</strong>
          <span role="status">
            {t(
              !available && config
                ? "chatNotEnabled"
                : busy
                ? "streaming"
                : session?.status === "error"
                ? "sessionUnavailable"
                : ready
                ? "sessionReady"
                : "loadingSession",
            )}
          </span>
        </div>
        <button onClick={startNewSession} disabled={!ready || busy}>
          <MessageSquarePlus size={15} />
          {t("newConversation")}
        </button>
      </div>
      <div className="messages">
        {!tab.messages.length && (
          <div className="chat-empty">
            <div className="agent-logo" aria-hidden="true" />
            <h1>{t(proactive ? "paChat" : "chatTitle")}</h1>
            <p>
              {t(
                IS_DEMO
                  ? "demoChatDescription"
                  : proactive
                  ? "paChatDescription"
                  : "chatDescription",
              )}
            </p>
            {suggestions}
          </div>
        )}
        {tab.messages.map((message) => (
          <div
            className={`message ${message.role}${
              message.source === "proactive" ? " proactive" : ""
            }`}
            key={message.id}
          >
            <div className="avatar">
              {message.role === "user" ? (
                t("you")
              ) : (
                <span
                  className="assistant-logo"
                  aria-label="ReMe"
                  title="ReMe"
                />
              )}
            </div>
            <div className="bubble">
              {message.source === "proactive" && (
                <span className="proactive-label">
                  prome · {t("proactiveMessage")}
                </span>
              )}
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
                (message.status === "interrupted" ||
                  message.status === "waiting") && (
                  <p role="status">
                    {t(
                      message.status === "interrupted"
                        ? "interrupted"
                        : "waiting",
                    )}
                  </p>
                )}
              {message.role === "assistant" &&
                tab.streaming &&
                !message.blocks?.some((block) => block.type !== "usage") && (
                  <LoaderCircle className="spin" size={16} />
                )}
            </div>
          </div>
        ))}
        {IS_DEMO && tab.messages.length > 0 && suggestions}
        <div ref={endRef} />
      </div>
      {!available && config && (
        <div className="session-sync-error" role="status">
          {t(proactive ? "paChatSetup" : "chatNotEnabled")}
        </div>
      )}
      {syncError && (
        <div className="session-sync-error" role="status">
          {t("sessionSyncFailed")}: {syncError}
        </div>
      )}
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
            disabled={!available}
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
            disabled={!input.trim() || !canSend}
            aria-label={t("send")}
          >
            <Send size={17} />
          </button>
        </div>
        <span>{t(IS_DEMO ? "demoChatHint" : "composerHint")}</span>
      </div>
    </div>
  );
}
