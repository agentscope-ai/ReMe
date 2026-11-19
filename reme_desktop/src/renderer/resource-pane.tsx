import { useState, type ReactNode } from "react";
import {
  ArrowDownLeft,
  ArrowLeft,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Inbox,
  Sparkles,
} from "lucide-react";
import type {
  DesktopState,
  DisplayMessage,
  ItemTarget,
  EventState,
} from "../shared/protocol.ts";
import { translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import { formatDate } from "./resources.ts";
import ContextMenu from "./context-menu.tsx";
import MessageBody from "./message-body.tsx";
import SplitPane from "./split-pane.tsx";

export interface ResourceRow {
  id: string;
  title: string;
  preview: string;
  time?: string | null;
  badge?: string;
  state?: Exclude<EventState, "all">;
  unread?: boolean;
  target?: ItemTarget;
}

export default function ResourcePane({
  state,
  rows,
  selected,
  onSelect,
  controls,
  pagination,
  children,
}: {
  state: DesktopState;
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
  children?: ReactNode;
}) {
  const t = translator(state.settings.language);
  const [menu, setMenu] = useState<{
    target: ItemTarget;
    x: number;
    y: number;
  } | null>(null);
  const [error, setError] = useState<string>();
  return (
    <SplitPane
      name="resources"
      className={`resource-browser ${selected ? "has-selection" : ""}`}
      language={state.settings.language}
      defaultWidth={300}
      minWidth={240}
      maxWidth={440}
      resizeLabel={t("resizeResourceList")}
      sidebar={
        <div className="resource-list-pane">
          {controls && <div className="resource-controls">{controls}</div>}
          {error && (
            <p className="action-error" role="alert">
              {error}
            </p>
          )}
          <div className="resource-list">
            {rows.length ? (
              rows.map((row) => (
                <button
                  key={row.id}
                  className={`resource-row ${selected === row.id ? "selected" : ""}`}
                  aria-pressed={selected === row.id}
                  onClick={() => onSelect(row.id)}
                  onContextMenu={(event) => {
                    if (row.target) {
                      event.preventDefault();
                      setMenu({
                        target: row.target,
                        x: event.clientX,
                        y: event.clientY,
                      });
                    }
                  }}
                >
                  <div className="resource-row-meta">
                    <span className="resource-sender">
                      {row.unread && <i className="unread-dot" />}
                      {row.title}
                    </span>
                    <time>{formatDate(row.time, state.settings.language)}</time>
                  </div>
                  <p>{row.preview}</p>
                  <div className="resource-row-footer">
                    <span data-state={row.state}>{row.badge}</span>
                  </div>
                </button>
              ))
            ) : (
              <div className="resource-empty">
                <Inbox size={30} strokeWidth={1.3} />
                <p>{t("noUpdates")}</p>
              </div>
            )}
          </div>
          {pagination && pagination.total > pagination.limit && (
            <nav className="resource-pagination" aria-label={t("navigation")}>
              <button
                className="icon-button"
                disabled={!pagination.offset}
                aria-label={t("inboxPrevious")}
                onClick={() =>
                  pagination.onPage(
                    Math.max(0, pagination.offset - pagination.limit),
                  )
                }
              >
                <ChevronLeft size={16} />
              </button>
              <span>
                {Math.floor(pagination.offset / pagination.limit) + 1} /{" "}
                {Math.max(1, Math.ceil(pagination.total / pagination.limit))}
              </span>
              <button
                className="icon-button"
                disabled={
                  pagination.offset + pagination.limit >= pagination.total
                }
                aria-label={t("inboxNext")}
                onClick={() =>
                  pagination.onPage(pagination.offset + pagination.limit)
                }
              >
                <ChevronRight size={16} />
              </button>
            </nav>
          )}
        </div>
      }
    >
      <div className="resource-detail">
        {selected && (
          <button className="resource-back" onClick={() => onSelect("")}>
            <ArrowLeft size={15} />
            {t("backToList")}
          </button>
        )}
        {children ?? (
          <div className="resource-empty">
            <ArrowDownLeft size={32} strokeWidth={1.2} />
            <h3>{t("selectResource")}</h3>
            <p>{t("selectResourceHint")}</p>
          </div>
        )}
      </div>
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
    </SplitPane>
  );
}

export function ResourceMessage({
  state,
  message,
  sourcePath,
  target,
  children,
}: {
  state: DesktopState;
  message: DisplayMessage;
  sourcePath?: string;
  target?: ItemTarget;
  children?: ReactNode;
}) {
  const t = translator(state.settings.language);
  const [error, setError] = useState<string>();
  const run = (value: Parameters<typeof action>[0]) =>
    void action(value).catch((reason) => setError(String(reason)));
  return (
    <article className="resource-article">
      <div className="resource-article-heading">
        <Sparkles size={18} />
        <time>{formatDate(message.createdAt, state.settings.language)}</time>
      </div>
      {children}
      {error && (
        <p className="action-error" role="alert">
          {error}
        </p>
      )}
      <MessageBody
        message={message}
        language={state.settings.language}
        onLink={(url) => run({ type: "link.open", url, sourcePath })}
      />
      {target && (
        <div className="resource-actions">
          {sourcePath && (
            <button
              className="subtle"
              onClick={() => run({ type: "document.open", path: sourcePath })}
            >
              {t("openInKnowledge")}
              <ArrowUpRight size={14} />
            </button>
          )}
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
            <ArrowUpRight size={14} />
          </button>
        </div>
      )}
    </article>
  );
}
