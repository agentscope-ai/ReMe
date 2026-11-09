import { useEffect, useId, useRef, useState } from "react";
import {
  ArrowUpRight,
  Bell,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileText,
  Inbox,
  Sparkles,
} from "lucide-react";
import type {
  DesktopState,
  ProactiveInboxItem,
  ProactiveInboxPage,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import { inboxDate, inboxRequest } from "./inbox.ts";

const PAGE_SIZE = 5;

function TaskFilter({
  label,
  allTasks,
  taskIds,
  value,
  onChange,
}: {
  label: string;
  allTasks: string;
  taskIds: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  const id = useId();
  const root = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const options = Array.from(new Set(["", value, ...taskIds]));
  const selected = Math.max(0, options.indexOf(value));
  const activeOption = Math.min(active, options.length - 1);
  const choose = (index: number) => {
    setOpen(false);
    onChange(options[index]);
  };
  useEffect(() => {
    if (!open) return;
    const close = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  useEffect(() => {
    if (open)
      root.current
        ?.querySelector('[data-active="true"]')
        ?.scrollIntoView({ block: "nearest" });
  }, [open, activeOption, taskIds]);
  return (
    <div
      className="task-filter"
      ref={root}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <button
        className="task-filter-trigger"
        role="combobox"
        aria-label={label}
        aria-expanded={open}
        aria-haspopup="listbox"
        aria-controls={id}
        aria-activedescendant={open ? `${id}-${activeOption}` : undefined}
        onClick={() => {
          setActive(selected);
          setOpen(!open);
        }}
        onKeyDown={(event) => {
          if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
            event.preventDefault();
            setOpen(true);
            setActive(
              event.key === "Home"
                ? 0
                : event.key === "End"
                  ? options.length - 1
                  : !open
                    ? selected
                    : (activeOption +
                        (event.key === "ArrowDown" ? 1 : -1) +
                        options.length) %
                      options.length,
            );
          } else if (open && (event.key === "Enter" || event.key === " ")) {
            event.preventDefault();
            choose(activeOption);
          } else if (event.key === "Escape") {
            event.preventDefault();
            setOpen(false);
          } else if (event.key === "Tab") setOpen(false);
        }}
      >
        <span>{value || allTasks}</span>
        <ChevronDown size={13} aria-hidden />
      </button>
      {open && (
        <div
          className="task-filter-menu"
          id={id}
          role="listbox"
          aria-label={label}
        >
          {options.map((option, index) => (
            <div
              key={option}
              id={`${id}-${index}`}
              className="task-filter-option"
              role="option"
              aria-selected={option === value}
              data-active={index === activeOption}
              onPointerMove={() => setActive(index)}
              onMouseDown={(event) => event.preventDefault()}
              onClick={() => choose(index)}
            >
              <span>{option || allTasks}</span>
              {option === value && <Check size={13} aria-hidden />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

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
  const [view, setView] = useState<"reminders" | "results">("reminders");
  const [chatLimit, setChatLimit] = useState(PAGE_SIZE);
  const [task, setTask] = useState("");
  const [offset, setOffset] = useState(0);
  const [page, setPage] = useState<ProactiveInboxPage | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [retry, setRetry] = useState(0);
  const [loading, setLoading] = useState(false);
  const read = new Set(state.readMessageIds);
  const messages = state.inboxMessages.toReversed();
  const reminders = view === "reminders";
  const locale = language === "zh" ? "zh-CN" : "en";
  let previousDay = "";
  const dayLabel = (timestamp: string | null | undefined) => {
    const date = timestamp ? new Date(timestamp) : null;
    return date && !Number.isNaN(date.getTime())
      ? date.toDateString() === new Date().toDateString()
        ? t("today")
        : date.toLocaleDateString(locale, { month: "short", day: "numeric" })
      : t("earlier");
  };

  useEffect(() => {
    if (state.focusMessageId) {
      setView("reminders");
      const index = state.inboxMessages
        .toReversed()
        .findIndex((item) => item.id === state.focusMessageId);
      setChatLimit((current) =>
        Math.max(current, Math.ceil((index + 1) / PAGE_SIZE) * PAGE_SIZE),
      );
    }
  }, [state.focusRequest, state.focusMessageId]);

  useEffect(() => {
    if (
      reminders ||
      state.activePage !== "remy" ||
      state.connection !== "connected"
    )
      return;
    let live = true;
    let pending = false;
    const refresh = async () => {
      if (pending) return;
      pending = true;
      setLoading(true);
      try {
        const value = await inboxRequest<ProactiveInboxPage>({
          type: "proactiveInbox",
          kind: "task_result",
          filter: "all",
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
    reminders,
    task,
    offset,
    retry,
    revision,
    state.activePage,
    state.connection,
  ]);

  return (
    <>
      <div className="sidebar-heading">
        <span className="eyebrow">REMY / INBOX</span>
        <h1>
          {t("remyUpdates")}
          {state.unread > 0 && (
            <span className="count-pill">{state.unread}</span>
          )}
        </h1>
        <p>{t(reminders ? "inboxCaption" : "taskInboxCaption")}</p>
      </div>
      <div className="source-tabs" role="group" aria-label={t("remyUpdates")}>
        {(["reminders", "results"] as const).map((value) => (
          <button
            key={value}
            className={view === value ? "active" : ""}
            aria-pressed={view === value}
            onClick={() => setView(value)}
          >
            {value === "reminders" ? <Bell size={13} /> : <Inbox size={13} />}
            {t(value === "reminders" ? "reminders" : "inbox")}
          </button>
        ))}
      </div>
      {!reminders && Boolean(page?.task_ids.length || task) && (
        <TaskFilter
          label={t("filterTask")}
          allTasks={t("allTasks")}
          taskIds={page?.task_ids ?? []}
          value={task}
          onChange={(value) => {
            setTask(value);
            setOffset(0);
          }}
        />
      )}
      <div className="inbox-list" aria-busy={!reminders && loading}>
        {reminders ? (
          <>
            {!messages.length && (
              <div className="sidebar-empty">
                <Bell size={26} strokeWidth={1.2} />
                <h3>{t("noUpdates")}</h3>
                <p>{t("noUpdatesHint")}</p>
              </div>
            )}
            {messages.slice(0, chatLimit).map((message) => {
              const day = dayLabel(message.createdAt);
              const group = day !== previousDay;
              previousDay = day;
              return (
                <div key={message.id}>
                  {group && <div className="list-section-label">{day}</div>}
                  <button
                    className={`inbox-card ${!read.has(message.id) ? "unread" : ""} ${!selected && state.focusMessageId === message.id ? "selected" : ""}`}
                    onClick={() =>
                      void action({ type: "chat.open", messageId: message.id })
                    }
                  >
                    <div className="inbox-card-top">
                      <Sparkles size={14} />
                      <span>Remy</span>
                      <time>{inboxDate(message.createdAt, language)}</time>
                    </div>
                    <p>{message.content}</p>
                    <div className="inbox-card-bottom">
                      <span
                        className="inbox-read-state"
                        data-unread={!read.has(message.id)}
                      >
                        {t(
                          read.has(message.id)
                            ? "inboxUserRead"
                            : "inboxUserUnread",
                        )}
                      </span>
                      <span className="inbox-card-link">
                        {t("conversation")}
                        <ArrowUpRight size={12} />
                      </span>
                    </div>
                  </button>
                </div>
              );
            })}
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
          </>
        ) : (
          <>
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
            {page && !page.items.length && error == null && page.complete && (
              <div className="sidebar-empty">
                <Inbox size={26} strokeWidth={1.2} />
                <h3>{t("noTaskResults")}</h3>
              </div>
            )}
            {page?.items.map((item) => {
              const day = dayLabel(item.created_at);
              const group = day !== previousDay;
              previousDay = day;
              return (
                <div key={item.id}>
                  {group && <div className="list-section-label">{day}</div>}
                  <button
                    className={`inbox-card ${!item.user_read_at ? "unread" : ""} ${selected === item.id ? "selected" : ""}`}
                    onClick={() => onOpen(item)}
                  >
                    <div className="inbox-card-top">
                      <FileText size={14} />
                      <span>{item.task_id}</span>
                    </div>
                    <h3 className="inbox-card-title">{item.title}</h3>
                    <p>{item.preview}</p>
                    <div className="inbox-card-bottom">
                      <time>{inboxDate(item.created_at, language)}</time>
                      <span
                        className="inbox-read-state"
                        data-unread={!item.user_read_at}
                      >
                        {t(
                          item.user_read_at
                            ? "inboxUserRead"
                            : "inboxUserUnread",
                        )}
                      </span>
                    </div>
                  </button>
                </div>
              );
            })}
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
          </>
        )}
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
