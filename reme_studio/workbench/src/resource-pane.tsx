import { useState, type ReactNode } from "react";
import {
  ArrowUpRight,
  Bell,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  FileText,
  Inbox,
  Send,
} from "lucide-react";
import type {
  StudioState,
  DisplayMessage,
  ItemTarget,
} from "../../client/src/protocol.ts";
import { translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import { formatDate, plainText } from "./resources.ts";
import ContextMenu from "./context-menu.tsx";
import MessageBody, { splitReminderSource } from "./message-body.tsx";
import {
  TimelineHeading,
  TimelineList,
  TimelinePane,
  type TimelineItem,
} from "./timeline.tsx";

export interface ResourceRow extends TimelineItem {
  target?: ItemTarget;
}

export function resourceSummary(content: string, fallback: string) {
  const lead = content.trimStart().split(/\r?\n[ \t]*\r?\n/, 1)[0];
  return plainText(lead) || fallback;
}

/** Promote a plain, standalone first line without hiding formatted content. */
export function resourceDetail(content: string, fallback: string) {
  const text = content.trimStart();
  const lead = /^(?:#{1,6} +)?([^\r\n]{1,72})\r?\n(?:[ \t]*\r?\n)+/.exec(text);
  if (
    lead &&
    plainText(lead[1]) === lead[1] &&
    text.slice(lead[0].length).trim()
  )
    return { title: lead[1], content: text.slice(lead[0].length) };
  return { title: fallback, content };
}

export default function ResourcePane({
  state,
  view,
  rows,
  selected,
  onSelect,
  controls,
  pagination,
  listKey,
  loading,
  children,
  embedded = false,
}: {
  state: StudioState;
  view: "reminders" | "inbox" | "outbox";
  rows: ResourceRow[];
  selected: string;
  onSelect: (id: string) => void;
  controls?: ReactNode;
  pagination?: {
    offset: number;
    limit: number;
    total: number;
    onPage: (offset: number) => void;
  };
  listKey?: string;
  loading?: boolean;
  children?: ReactNode;
  embedded?: boolean;
}) {
  const { action } = useHost();
  const language = state.settings.language;
  const t = translator(language);
  const [menu, setMenu] = useState<{
    target: ItemTarget;
    x: number;
    y: number;
  } | null>(null);
  const [error, setError] = useState<string>();
  const Icon = view === "reminders" ? Bell : view === "inbox" ? Inbox : Send;
  return (
    <>
      {error && (
        <p className="action-error" role="alert">
          {error}
        </p>
      )}
      <TimelinePane
        name="resources"
        language={language}
        listKey={listKey}
        detailKey={selected}
        onClose={() => onSelect("")}
        heading={
          embedded ? (
            <div className="resource-embedded-controls">
              <div className="resource-controls">{controls}</div>
              <span>
                {t("flowItemCount", {
                  count: pagination?.total ?? rows.length,
                })}
              </span>
            </div>
          ) : (
            <TimelineHeading
              title={t(view)}
              icon={<Icon size={22} />}
              controls={
                <>
                  <div className="resource-controls">{controls}</div>
                  <span className="timeline-order">
                    <CalendarDays size={14} aria-hidden />
                    {t("timelineNewest")}
                  </span>
                </>
              }
            />
          )
        }
        list={
          <>
            {rows.length ? (
              <TimelineList
                key={listKey}
                items={rows}
                language={language}
                selected={selected}
                onSelect={onSelect}
                holdNew={
                  pagination?.offset === undefined || pagination.offset === 0
                }
                onItemMenu={(id, x, y) => {
                  const target = rows.find((row) => row.id === id)?.target;
                  if (target) setMenu({ target, x, y });
                  return Boolean(target);
                }}
              />
            ) : (
              <p className="timeline-empty">
                {t(loading ? "documentLoading" : "noUpdates")}
              </p>
            )}
            {pagination && pagination.total > pagination.limit && (
              <nav className="timeline-pagination" aria-label={t("navigation")}>
                <button
                  type="button"
                  className="icon-button"
                  disabled={!pagination.offset}
                  aria-label={t("inboxPrevious")}
                  onClick={() =>
                    pagination.onPage(
                      Math.max(0, pagination.offset - pagination.limit),
                    )
                  }
                >
                  <ChevronLeft size={16} aria-hidden />
                </button>
                <span>
                  {t("pageLabel", {
                    page: Math.floor(pagination.offset / pagination.limit) + 1,
                    total: Math.ceil(pagination.total / pagination.limit),
                  })}
                </span>
                <button
                  type="button"
                  className="icon-button"
                  disabled={
                    pagination.offset + pagination.limit >= pagination.total
                  }
                  aria-label={t("inboxNext")}
                  onClick={() =>
                    pagination.onPage(pagination.offset + pagination.limit)
                  }
                >
                  <ChevronRight size={16} aria-hidden />
                </button>
              </nav>
            )}
          </>
        }
      >
        {selected &&
          (children ?? (
            <p className="timeline-empty">
              {t(loading ? "documentLoading" : "itemUnavailable")}
            </p>
          ))}
      </TimelinePane>
      {menu && (
        <ContextMenu
          position={{ x: menu.x, y: menu.y }}
          onClose={() => setMenu(null)}
          items={(
            [
              ["open", "viewDetails"],
              ["remind", "remindAgain"],
              ["explain", "explainReminder"],
              ["copy", "copyContent"],
            ] as const
          ).map(([id, label]) => ({
            id,
            label: t(label),
            disabled:
              id === "explain" &&
              (state.connection !== "connected" ||
                state.status === "active" ||
                !!state.pending?.streaming),
            select: () =>
              void action({
                type: "item.action",
                action: id,
                target: menu.target,
              }).catch((reason) => setError(String(reason))),
          }))}
        />
      )}
    </>
  );
}

export function ResourceMessage({
  state,
  message,
  sourcePath,
  target,
  sender,
  metadata,
}: {
  state: StudioState;
  message: DisplayMessage;
  sourcePath?: string;
  target?: ItemTarget;
  sender: string;
  metadata?: ReactNode;
}) {
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [error, setError] = useState<string>();
  const source = splitReminderSource(message, state.clientInfo?.workspace_dir);
  const paths = [...new Set([source?.path, sourcePath])].filter(
    (path): path is string => Boolean(path),
  );
  const run = (value: Parameters<typeof action>[0]) =>
    void action(value).catch((reason) => setError(String(reason)));
  return (
    <article className="detail-article">
      <header className="detail-meta resource-message-meta">
        {sender && <span>{sender}</span>}
        {metadata}
        {message.createdAt && (
          <time dateTime={message.createdAt}>
            {formatDate(message.createdAt, state.settings.language)}
          </time>
        )}
      </header>
      {error && (
        <p className="action-error" role="alert">
          {error}
        </p>
      )}
      <MessageBody
        message={source?.display ?? message}
        language={state.settings.language}
        onLink={(url) => run({ type: "link.open", url, sourcePath })}
      />
      {(paths.length > 0 || target) && (
        <footer className="detail-section resource-actions">
          {paths.map((path) => (
            <a
              key={path}
              className="subtle resource-source-link"
              href={`#reme-file=${encodeURIComponent(path)}`}
              aria-label={t("openReminderSource", { path })}
              title={path}
              onClick={(event) => {
                event.preventDefault();
                run({ type: "document.open", path });
              }}
            >
              <FileText size={14} aria-hidden />
              <span>{path.split(/[\\/]/).at(-1) || path}</span>
            </a>
          ))}
          {target && (
            <button
              className="subtle"
              disabled={
                state.connection !== "connected" ||
                state.status === "active" ||
                !!state.pending?.streaming
              }
              onClick={() =>
                run({ type: "item.action", action: "explain", target })
              }
            >
              {t("explainReminder")}
              <ArrowUpRight size={14} aria-hidden />
            </button>
          )}
        </footer>
      )}
    </article>
  );
}
