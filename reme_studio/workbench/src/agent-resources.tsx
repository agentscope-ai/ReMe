import { useEffect, useState } from "react";
import { ArrowUpRight, ListFilter, RefreshCw } from "lucide-react";
import type {
  TaskReminder,
  StudioState,
  EventState,
  InputEvent,
  WorkspaceRequest,
} from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import {
  formatDate,
  plainText,
  useResource,
  resourceSource,
  useResourcePage,
} from "./resources.ts";
import ResourcePane, { ResourceMessage } from "./resource-pane.tsx";

export default function AgentResources({
  state,
  inbox,
}: {
  state: StudioState;
  inbox: boolean;
}) {
  const requestResource = useResource();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const agentName = state.selectedAgentName ?? "Remy";
  const active = state.activePage === (agentName === "Remy" ? "remy" : "tasks");
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<EventState>("all");
  const [selected, setSelected] = useState("");
  const [focusedMail, setFocusedMail] = useState<TaskReminder>();
  const [readError, setReadError] = useState<unknown>(null);
  const request: WorkspaceRequest = inbox
    ? { type: "inbox", agentName, offset, state: filter }
    : {
        type: "outbox",
        agentName,
        offset,
        state: filter === "processing" ? "pending" : filter,
      };
  const { page, error, loading, refresh } = useResourcePage<
    TaskReminder | InputEvent
  >(request, active && state.connection === "connected");
  useEffect(() => {
    if (
      state.focusTarget?.kind === "mail" &&
      state.focusTarget.agentName === agentName
    )
      setSelected(state.focusTarget.id);
  }, [state.focusTarget, state.focusRequest, agentName, inbox]);
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
    if (
      !selected ||
      !page ||
      !active ||
      (inbox && agentName !== "Remy") ||
      Boolean(listed)
    )
      return;
    let live = true;
    void requestResource<TaskReminder>({
      type: "reminder",
      agentName,
      id: selected,
    })
      .then((value) => {
        if (live) setFocusedMail(value);
      })
      .catch((reason) => {
        if (live) setReadError(reason);
      });
    return () => {
      live = false;
    };
  }, [inbox, selected, agentName, Boolean(listed), Boolean(page), active]);
  const item =
    listed && "event" in listed
      ? (listed.reminder ?? listed)
      : (listed ?? (focusedMail?.id === selected ? focusedMail : undefined));
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
                    ...(inbox ? ["processing"] : []),
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
        rows={(page?.items ?? []).map((value) => {
          if ("event" in value && !value.reminder)
            return {
              id: value.event.id,
              title: resourceSource(
                value.event.source,
                state.settings.language,
              ),
              preview: value.event.description,
              time: value.event.event_ts,
              badge: stateLabel(value.state),
              state: value.state,
            };
          const mail = "event" in value ? value.reminder! : value;
          const status =
            "event" in value
              ? value.state
              : mail.read_at
                ? "processed"
                : "pending";
          return {
            id: mail.id,
            title: mail.from_agent_name,
            preview: plainText(mail.message),
            time: mail.delivered_at,
            badge: `${mail.priority} · ${stateLabel(status)}`,
            state: status,
            target: { kind: "mail" as const, agentName, id: mail.id },
          };
        })}
      >
        {item &&
          ("event" in item ? (
            <article className="resource-article">
              <div className="resource-event-meta">
                {stateLabel(item.state)} ·{" "}
                {formatDate(item.event.event_ts, state.settings.language)}
              </div>
              <h2 className="resource-title">
                {resourceSource(item.event.source, state.settings.language)}
              </h2>
              <p className="resource-description">{item.event.description}</p>
              {item.consume_ts && (
                <p className="resource-timestamp">
                  {t("processedAt")}{" "}
                  {formatDate(item.consume_ts, state.settings.language)}
                </p>
              )}
              <details className="resource-files">
                <summary>
                  {t("relatedFiles")}{" "}
                  <span className="count-badge">{item.event.paths.length}</span>
                </summary>
                <div className="resource-links">
                  {item.event.paths.map((path) => (
                    <button
                      key={path}
                      className="resource-path"
                      title={path}
                      onClick={() => open(path)}
                    >
                      {path.split(/[\\/]/).at(-1) || path}
                      <ArrowUpRight size={14} />
                    </button>
                  ))}
                </div>
              </details>
              <details className="resource-original">
                <summary>{t("originalInformation")}</summary>
                <p>{item.event.source}</p>
                {item.event.paths.map((path) => (
                  <p key={path}>{path}</p>
                ))}
              </details>
            </article>
          ) : (
            <ResourceMessage
              state={state}
              message={{
                id: item.id,
                role: "assistant",
                source: "proactive",
                content: item.message,
                createdAt: item.delivered_at,
                completed: true,
                streaming: false,
                status: null,
              }}
              sourcePath={item.id}
              target={{ kind: "mail", agentName, id: item.id }}
            >
              <div className="mail-receipts">
                <span className="priority-pill">{item.priority}</span>
                <span>
                  {t("fromAgent")} {item.from_agent_name}
                </span>
                <span>
                  {stateLabel(item.read_at ? "processed" : "pending")}
                </span>
              </div>
            </ResourceMessage>
          ))}
      </ResourcePane>
    </>
  );
}
