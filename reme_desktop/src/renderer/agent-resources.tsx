import { useEffect, useState } from "react";
import { ArrowUpRight, ListFilter, RefreshCw } from "lucide-react";
import type {
  AgentMail,
  DesktopState,
  EventState,
  InputEvent,
  WorkspaceRequest,
} from "../shared/protocol.ts";
import { errorText, translator } from "../shared/i18n.ts";
import { action } from "./desktop.ts";
import {
  formatDate,
  plainText,
  requestResource,
  useResourcePage,
} from "./resources.ts";
import ResourcePane, { ResourceMessage } from "./resource-pane.tsx";
import { watchReadReceipt } from "./read-receipt.ts";

export default function AgentResources({
  state,
  events,
}: {
  state: DesktopState;
  events: boolean;
}) {
  const t = translator(state.settings.language);
  const agentId = state.selectedAgentId ?? "chat";
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<EventState>("all");
  const [selected, setSelected] = useState("");
  const [focusedMail, setFocusedMail] = useState<AgentMail>();
  const [readError, setReadError] = useState<unknown>(null);
  const request: WorkspaceRequest = events
    ? { type: "events", agentId, offset, state: filter }
    : {
        type: "mailbox",
        agentId,
        offset,
        state: filter === "processing" ? "pending" : filter,
      };
  const { page, error, loading, refresh } = useResourcePage<
    AgentMail | InputEvent
  >(
    request,
    `${state.settings.endpoint}:${state.workspace}`,
    state.activePage === "remy" && state.connection === "connected",
    state.mailboxRevision,
  );
  useEffect(() => {
    if (
      !events &&
      state.focusTarget?.kind === "mail" &&
      state.focusTarget.agentId === agentId
    )
      setSelected(state.focusTarget.id);
  }, [state.focusTarget, state.focusRequest, agentId, events]);
  useEffect(() => {
    if (page && offset && offset >= page.total)
      setOffset(Math.max(0, Math.ceil(page.total / 20) - 1) * 20);
  }, [page, offset]);
  const listed = page?.items.find(
    (value) => ("event" in value ? value.event.id : value.id) === selected,
  );
  useEffect(() => {
    setFocusedMail(undefined);
    setReadError(null);
    if (events || !selected || listed || !page || state.activePage !== "remy")
      return;
    let live = true;
    void requestResource<AgentMail>({ type: "mail", agentId, id: selected })
      .then((value) => {
        if (live) setFocusedMail(value);
      })
      .catch((reason) => {
        if (live) setReadError(reason);
      });
    return () => {
      live = false;
    };
  }, [
    events,
    selected,
    agentId,
    Boolean(listed),
    Boolean(page),
    state.activePage,
  ]);
  const item = listed ?? focusedMail;
  const mail = item && !("event" in item) ? item : undefined;
  useEffect(() => {
    if (
      !mail ||
      mail.user_read_at ||
      !state.chatVisible ||
      !state.chatFocused ||
      state.activePage !== "remy"
    )
      return;
    const receipt = watchReadReceipt(async () => {
      await requestResource({
        type: "mail",
        agentId,
        id: mail.id,
        markRead: true,
      });
      return true;
    }, setReadError);
    return receipt.close;
  }, [
    mail?.id,
    mail?.user_read_at,
    agentId,
    state.chatVisible,
    state.chatFocused,
    state.activePage,
  ]);
  const stateLabel = (value: string) =>
    t(
      value === "pending"
        ? "resourcePending"
        : value === "processing"
          ? "resourceProcessing"
          : value === "processed"
            ? "resourceProcessed"
            : "resourceAll",
    );
  const open = (path: string) =>
    void action({ type: "document.open", path }).catch(setReadError);
  return (
    <>
      {(error ?? readError) != null && (
        <p className="action-error" role="alert">
          {errorText(error ?? readError, state.settings.language)}{" "}
          <button className="subtle" onClick={refresh}>
            {t("retry")}
          </button>
        </p>
      )}
      <ResourcePane
        state={state}
        selected={selected}
        onSelect={setSelected}
        controls={
          <>
            <label className="resource-filter">
              <ListFilter size={14} aria-hidden />
              <select
                value={filter}
                aria-label={t("resourceFilter")}
                onChange={(event) => {
                  setFilter(event.target.value as EventState);
                  setOffset(0);
                  setSelected("");
                }}
              >
                {(
                  [
                    "all",
                    "pending",
                    ...(events ? ["processing"] : []),
                    "processed",
                  ] as EventState[]
                ).map((value) => (
                  <option key={value} value={value}>
                    {value === "all"
                      ? t("resourceAllStates")
                      : stateLabel(value)}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="icon-button"
              onClick={refresh}
              aria-label={t("refreshFiles")}
            >
              <RefreshCw size={14} className={loading ? "spinning" : ""} />
            </button>
          </>
        }
        pagination={
          page
            ? {
                offset,
                limit: page.limit,
                total: page.total,
                onPage: (value) => {
                  setOffset(value);
                  setSelected("");
                },
              }
            : undefined
        }
        rows={(page?.items ?? []).map((value) =>
          "event" in value
            ? {
                id: value.event.id,
                title: value.event.source,
                preview: value.event.description,
                time: value.event.event_ts,
                badge: stateLabel(value.state),
                state: value.state,
              }
            : {
                id: value.id,
                title: value.from_agent_id,
                preview: plainText(value.message),
                time: value.created_at,
                badge: `${value.priority} · ${stateLabel(value.read_at ? "processed" : "pending")}`,
                unread: !value.user_read_at,
                state: value.read_at ? "processed" : "pending",
                target: { kind: "mail", agentId, id: value.id },
              },
        )}
      >
        {item &&
          ("event" in item ? (
            <article className="resource-article">
              <div className="eyebrow">
                {item.event.source} · {stateLabel(item.state)}
              </div>
              <h2>{item.event.description}</h2>
              <p className="resource-timestamp">
                {formatDate(item.event.event_ts, state.settings.language)}
              </p>
              {item.consume_ts && (
                <p className="resource-timestamp">
                  {t("processedAt")}{" "}
                  {formatDate(item.consume_ts, state.settings.language)}
                </p>
              )}
              <div className="resource-links">
                {item.event.paths.map((path) => (
                  <button
                    key={path}
                    className="resource-path"
                    onClick={() => open(path)}
                  >
                    {path}
                    <ArrowUpRight size={14} />
                  </button>
                ))}
              </div>
            </article>
          ) : (
            <ResourceMessage
              state={state}
              message={{
                id: item.id,
                role: "assistant",
                source: "proactive",
                content: item.message,
                createdAt: item.created_at,
                completed: true,
                streaming: false,
                status: null,
              }}
              sourcePath={item.paths[0]}
              target={{ kind: "mail", agentId, id: item.id }}
            >
              <div className="mail-receipts">
                <span className="priority-pill">{item.priority}</span>
                <span>
                  {t("fromAgent")} {item.from_agent_id}
                </span>
                <span>
                  {t(item.delivered_at ? "mailDelivered" : "mailWaiting")}
                </span>
                <span>
                  {stateLabel(item.read_at ? "processed" : "pending")}
                </span>
              </div>
              <div className="resource-links">
                {item.paths.slice(1).map((path) => (
                  <button
                    key={path}
                    className="resource-path"
                    onClick={() => open(path)}
                  >
                    {path}
                    <ArrowUpRight size={14} />
                  </button>
                ))}
              </div>
            </ResourceMessage>
          ))}
      </ResourcePane>
    </>
  );
}
