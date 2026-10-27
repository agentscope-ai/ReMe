"use client";

import { useEffect, useRef, useState } from "react";
import { Check, CircleAlert, LoaderCircle, Send } from "lucide-react";
import WorkspaceMarkdown from "./files-workspace/WorkspaceMarkdown";
import { streamChat } from "./api";
import { chatStreamError, formatStreamPayloads } from "./chat-stream";
import { useI18n } from "./i18n";
import { useWorkspaceStore } from "./store";
import { IS_DEMO } from "./studio-mode";
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
  const failed = block.status === "error";
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
        <small>{active ? t("streaming") : t("completed")}</small>
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
}: {
  tab: Extract<WorkspaceTab, { type: "agent" }>;
}) {
  const { language, t } = useI18n();
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
    const requestController = new AbortController();
    controller.current = requestController;
    const error = await chatStreamError(
      () =>
        streamChat(query, tab.sessionId, requestController.signal, (chunk) =>
          applyChatChunk(tab.id, assistant.id, chunk),
        ),
      requestController.signal,
      t("chatFailed"),
    );
    finishChat(tab.id, assistant.id, error);
    if (controller.current === requestController) controller.current = null;
  };

  return (
    <div className="chat">
      <div className="messages">
        {!tab.messages.length && (
          <div className="chat-empty">
            <div className="agent-logo" aria-hidden="true" />
            <h1>{t("chatTitle")}</h1>
            <p>{t(IS_DEMO ? "demoChatDescription" : "chatDescription")}</p>
            {suggestions}
          </div>
        )}
        {tab.messages.map((message) => (
          <div className={`message ${message.role}`} key={message.id}>
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
        {IS_DEMO && tab.messages.length > 0 && suggestions}
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
        <span>{t(IS_DEMO ? "demoChatHint" : "composerHint")}</span>
      </div>
    </div>
  );
}
