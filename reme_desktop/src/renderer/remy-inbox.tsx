import { useEffect, useState } from "react";
import {
  ArrowUpRight,
  Bell,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileText,
  MessageSquare,
  Sparkles,
} from "lucide-react";
import type {
  DesktopState,
  InboxKind,
  ProactiveInboxItem,
  ProactiveInboxPage,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import { inboxDate, inboxRequest, inboxStatus } from "./inbox.ts";

const PAGE_SIZE = 5;

export default function RemyInbox({
  state,
  selected,
  revision,
  onOpen,
}: {
  state: DesktopState;
  selected: string;
  revision: number;
  onOpen: (item: ProactiveInboxItem) => void;
}) {
  const language = state.settings.language;
  const t = translator(language);
  const scope = `${state.settings.endpoint}:${state.workspace}`;
  const [chatOpen, setChatOpen] = useState(true);
  const [proactiveOpen, setProactiveOpen] = useState(true);
  const [chatFilter, setChatFilter] = useState("all");
  const [chatLimit, setChatLimit] = useState(PAGE_SIZE);
  const [kind, setKind] = useState<InboxKind | "all">("all");
  const [filter, setFilter] = useState<"all" | "pending" | "unread">("all");
  const [task, setTask] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ProactiveInboxPage | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [retry, setRetry] = useState(0);
  const [loading, setLoading] = useState(false);
  const read = new Set(state.readMessageIds);
  const messages = state.inboxMessages
    .filter((message) => chatFilter === "all" || !read.has(message.id))
    .toReversed();

  useEffect(() => {
    if (state.focusMessageId) {
      setChatOpen(true);
      setChatFilter("all");
      const index = state.inboxMessages
        .toReversed()
        .findIndex((item) => item.id === state.focusMessageId);
      setChatLimit((current) =>
        Math.max(current, Math.ceil((index + 1) / PAGE_SIZE) * PAGE_SIZE),
      );
    }
  }, [state.focusRequest, state.focusMessageId]);

  useEffect(() => {
    if (state.activePage !== "remy" || state.connection !== "connected") return;
    let live = true;
    let pending = false;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      setLoading(true);
      try {
        const value = await inboxRequest<ProactiveInboxPage>({
          type: "proactiveInbox",
          kind,
          filter,
          taskId: task || undefined,
          offset,
          limit: PAGE_SIZE,
        });
        if (!live) return;
        if (offset > 0 && offset >= value.total) {
          setOffset(
            Math.max(0, Math.ceil(value.total / PAGE_SIZE) - 1) * PAGE_SIZE,
          );
        } else {
          setPage(value);
          setError(null);
        }
      } catch (error) {
        if (live) setError(error);
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
  }, [
    scope,
    kind,
    filter,
    task,
    offset,
    retry,
    revision,
    state.activePage,
    state.connection,
  ]);

  const changeKind = (value: InboxKind | "all") => {
    if (kind === value) return;
    setKind(value);
    setTask("");
    setOffset(0);
    setPage(null);
    if (filter === "unread" && value !== "task_result") setFilter("all");
  };

  return (
    <>
      <div className="sidebar-heading">
        <span className="eyebrow">REMY / INBOX</span>
        <h1>{t("inbox")}</h1>
        <p>{t("inboxCaption")}</p>
      </div>
      <div className="inbox-sections">
        <section className="inbox-section" aria-labelledby="chat-inbox-heading">
          <button
            id="chat-inbox-heading"
            className="inbox-section-heading"
            aria-expanded={chatOpen}
            aria-controls="chat-inbox-content"
            onClick={() => setChatOpen(!chatOpen)}
          >
            <MessageSquare size={15} />
            <span>Chat Agent</span>
            <span className="count-pill">
              {state.unread} {t("unreadUpdates")}
            </span>
            <ChevronDown size={14} className={chatOpen ? "" : "collapsed"} />
          </button>
          {chatOpen && (
            <div id="chat-inbox-content">
              <div
                className="inbox-filters"
                role="group"
                aria-label={t("chatInboxFilter")}
              >
                {(["all", "unread"] as const).map((value) => (
                  <button
                    key={value}
                    className={chatFilter === value ? "active" : ""}
                    aria-pressed={chatFilter === value}
                    onClick={() => {
                      setChatFilter(value);
                      setChatLimit(PAGE_SIZE);
                    }}
                  >
                    {t(value === "all" ? "allUpdates" : "unreadUpdates")}
                  </button>
                ))}
              </div>
              <div className="inbox-list">
                {!messages.length && (
                  <div className="sidebar-empty">
                    <Bell size={23} strokeWidth={1.2} />
                    <h3>
                      {t(chatFilter === "unread" ? "allCaughtUp" : "noUpdates")}
                    </h3>
                    <p>{t("noUpdatesHint")}</p>
                  </div>
                )}
                {messages.slice(0, chatLimit).map((message) => (
                  <button
                    key={message.id}
                    className={`inbox-card ${!read.has(message.id) ? "unread" : ""} ${!selected && state.focusMessageId === message.id ? "selected" : ""}`}
                    onClick={() =>
                      void action({ type: "chat.open", messageId: message.id })
                    }
                  >
                    <div className="inbox-card-top">
                      <Sparkles size={14} />
                      <span>Remy</span>
                      <time>{inboxDate(message.createdAt, language)}</time>
                      {!read.has(message.id) && <i className="unread-dot" />}
                    </div>
                    <p>{message.content}</p>
                    <span className="inbox-card-link">
                      {t("conversation")}
                      <ArrowUpRight size={12} />
                    </span>
                  </button>
                ))}
                {messages.length > PAGE_SIZE && (
                  <button
                    className="inbox-more"
                    onClick={() =>
                      setChatLimit(
                        chatLimit < messages.length
                          ? chatLimit + PAGE_SIZE
                          : PAGE_SIZE,
                      )
                    }
                  >
                    {t(
                      chatLimit < messages.length
                        ? "inboxShowMore"
                        : "inboxShowLess",
                    )}
                  </button>
                )}
              </div>
            </div>
          )}
        </section>
        <section
          className="inbox-section"
          aria-labelledby="proactive-inbox-heading"
        >
          <button
            id="proactive-inbox-heading"
            className="inbox-section-heading"
            aria-expanded={proactiveOpen}
            aria-controls="proactive-inbox-content"
            onClick={() => setProactiveOpen(!proactiveOpen)}
          >
            <Sparkles size={15} />
            <span>Proactive Agent</span>
            <span className="count-pill" title={t("proactivePendingHint")}>
              {page?.complete && !error ? page.pending : "—"}{" "}
              {t("inboxPending")}
            </span>
            <ChevronDown
              size={14}
              className={proactiveOpen ? "" : "collapsed"}
            />
          </button>
          {proactiveOpen && (
            <div id="proactive-inbox-content">
              <div
                className="inbox-filters"
                role="group"
                aria-label={t("inboxKind")}
              >
                {(["all", "task_result", "instruction"] as const).map(
                  (value) => (
                    <button
                      key={value}
                      className={kind === value ? "active" : ""}
                      aria-pressed={kind === value}
                      onClick={() => changeKind(value)}
                    >
                      {t(
                        value === "all"
                          ? "allUpdates"
                          : value === "task_result"
                            ? "taskResults"
                            : "inboxInstructions",
                      )}
                    </button>
                  ),
                )}
              </div>
              <div
                className="inbox-filters inbox-status-filters"
                role="group"
                aria-label={t("proactiveInboxFilter")}
              >
                {(
                  [
                    "all",
                    "pending",
                    ...(kind === "task_result" ? ["unread" as const] : []),
                  ] as const
                ).map((value) => (
                  <button
                    key={value}
                    className={filter === value ? "active" : ""}
                    aria-pressed={filter === value}
                    onClick={() => {
                      if (filter !== value) {
                        setFilter(value);
                        setOffset(0);
                        setPage(null);
                      }
                    }}
                  >
                    {t(
                      value === "all"
                        ? "allUpdates"
                        : value === "pending"
                          ? "inboxPending"
                          : "inboxUserUnread",
                    )}
                  </button>
                ))}
              </div>
              {kind === "task_result" && (
                <select
                  className="task-filter"
                  value={task}
                  aria-label={t("filterTask")}
                  onChange={(event) => {
                    setTask(event.target.value);
                    setOffset(0);
                  }}
                >
                  <option value="">{t("allTasks")}</option>
                  {page?.task_ids.map((id) => (
                    <option key={id} value={id}>
                      {id}
                    </option>
                  ))}
                </select>
              )}
              <div className="inbox-list" aria-busy={loading}>
                {error != null && (
                  <div className="inbox-warning" role="alert">
                    <p>{errorText(error, language)}</p>
                    <button onClick={() => setRetry(retry + 1)}>
                      {t("inboxRetry")}
                    </button>
                  </div>
                )}
                {Boolean(page?.warnings.length) && (
                  <div className="inbox-warning" role="status">
                    <strong>{t("inboxPartial")}</strong>
                    <details>
                      <summary>{t("inboxDetails")}</summary>
                      {page?.warnings.map((warning, index) => (
                        <p key={index}>{warning}</p>
                      ))}
                    </details>
                  </div>
                )}
                {!page && error == null && (
                  <p className="inbox-empty">
                    {t(
                      state.connection === "connected"
                        ? "documentLoading"
                        : "reconnecting",
                    )}
                  </p>
                )}
                {page &&
                  !page.items.length &&
                  error == null &&
                  page.complete && (
                    <p className="inbox-empty">
                      {t(
                        filter === "pending"
                          ? "inboxNoPending"
                          : filter === "unread"
                            ? "inboxNoUnreadResults"
                            : "inboxNoInputs",
                      )}
                    </p>
                  )}
                {page?.items.map((item) => (
                  <button
                    key={item.id}
                    className={`inbox-card ${item.status === "pending" ? "awaiting" : ""} ${selected === item.id ? "selected" : ""}`}
                    onClick={() => onOpen(item)}
                  >
                    <div className="inbox-card-top">
                      {item.kind === "task_result" ? (
                        <FileText size={14} />
                      ) : (
                        <MessageSquare size={14} />
                      )}
                      <span>
                        {t(
                          item.kind === "task_result"
                            ? "taskResults"
                            : "inboxInstructions",
                        )}
                      </span>
                      <span className={`inbox-status ${item.status}`}>
                        {inboxStatus(item.status, language)}
                      </span>
                    </div>
                    <h3 className="inbox-card-title">{item.title}</h3>
                    <p>{item.preview}</p>
                    <div className="inbox-card-bottom">
                      <time>{inboxDate(item.created_at, language)}</time>
                      {item.kind === "task_result" && (
                        <span>
                          {t(
                            item.user_read_at
                              ? "inboxUserRead"
                              : "inboxUserUnread",
                          )}
                        </span>
                      )}
                    </div>
                  </button>
                ))}
                {page && page.total > PAGE_SIZE && (
                  <div className="inbox-pagination">
                    <button
                      aria-label={t("inboxPrevious")}
                      disabled={offset === 0 || loading}
                      onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                    >
                      <ChevronLeft size={15} />
                    </button>
                    <span>
                      {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} /{" "}
                      {page.total}
                    </span>
                    <button
                      aria-label={t("inboxNext")}
                      disabled={offset + PAGE_SIZE >= page.total || loading}
                      onClick={() => setOffset(offset + PAGE_SIZE)}
                    >
                      <ChevronRight size={15} />
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </div>
      <div className="sidebar-footer">
        <span
          className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
        />
        {t(state.connection === "connected" ? "connected" : "reconnecting")}
      </div>
    </>
  );
}
