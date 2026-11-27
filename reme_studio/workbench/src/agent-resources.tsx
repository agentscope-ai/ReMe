import { useEffect, useState } from "react";
import { ArrowUpRight, FileText, ListFilter, RefreshCw } from "lucide-react";
import type {
  TaskReminder,
  StudioState,
  EventState,
  InputEvent,
  ResourcePage,
  WorkspaceRequest,
} from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import {
  formatDate,
  useResource,
  resourceSource,
  usePollingResource,
} from "./resources.ts";
import ResourcePane, {
  ResourceMessage,
  resourceSummary,
  resourceDetail,
} from "./resource-pane.tsx";
import { DetailHeading, TimelineBadge } from "./timeline.tsx";

type ListedResource =
  InputEvent | (TaskReminder & { state: InputEvent["state"] });

export default function AgentResources({
  state,
  inbox,
  revision,
  agentName = state.selectedAgentName ?? "Remy",
  active = state.activePage === (agentName === "Remy" ? "remy" : "tasks"),
  embedded = false,
  agentLabel = (name: string) => name,
}: {
  state: StudioState;
  inbox: boolean;
  revision?: number;
  agentName?: string;
  active?: boolean;
  embedded?: boolean;
  agentLabel?: (name: string) => string;
}) {
  const requestResource = useResource();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [offset, setOffset] = useState(0);
  const [filter, setFilter] = useState<EventState>("all");
  const [selected, setSelected] = useState("");
  const [focusedMail, setFocusedMail] = useState<TaskReminder>();
  const [readError, setReadError] = useState<unknown>(null);
  const [readAttempt, setReadAttempt] = useState(0);
  const request: WorkspaceRequest = inbox
    ? { type: "inbox", agentName, offset, state: filter }
    : {
        type: "outbox",
        agentName,
        offset,
        state: filter === "processing" ? "pending" : filter,
      };
  const {
    data: page,
    error,
    loading,
    refresh,
  } = usePollingResource<ResourcePage<ListedResource>>(
    request,
    active && state.connection === "connected",
    revision,
  );
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
  }, [
    inbox,
    selected,
    agentName,
    Boolean(listed),
    Boolean(page),
    active,
    readAttempt,
  ]);
  const item =
    listed ??
    (focusedMail?.id === selected
      ? {
          ...focusedMail,
          state: focusedMail.read_at ? "processed" : "pending",
        }
      : undefined);
  const stateLabel = (value: string) => {
    if (!inbox && value === "pending") return t("resourceDelivered");
    if ((agentName === "Remy" || !inbox) && value === "processed")
      return t("resourceRemyProcessed");
    return t(
      value === "pending"
        ? "resourcePending"
        : value === "processing"
          ? "resourceProcessing"
          : value === "processed"
            ? "resourceProcessed"
            : "resourceAll",
    );
  };
  const open = (path: string) =>
    void action({ type: "document.open", path }).catch(setReadError);
  const eventSource =
    item && "event" in item
      ? resourceSource(item.event.source, state.settings.language)
      : "";
  const eventDetail =
    item && "event" in item
      ? resourceDetail(item.event.description, eventSource)
      : undefined;
  return (
    <>
      {(error ?? readError) != null && (
        <p className="action-error" role="alert">
          {errorText(error ?? readError, state.settings.language)}{" "}
          <button
            className="subtle"
            onClick={() => {
              refresh();
              setReadAttempt((attempt) => attempt + 1);
            }}
          >
            {t("retry")}
          </button>
        </p>
      )}
      <ResourcePane
        state={state}
        embedded={embedded}
        view={inbox ? "inbox" : "outbox"}
        listKey={`${filter}:${offset}`}
        loading={
          loading ||
          Boolean(
            selected && !item && !readError && (agentName === "Remy" || !inbox),
          )
        }
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
              aria-label={t(inbox ? "refreshInbox" : "refreshOutbox")}
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
          const badge = (
            <TimelineBadge
              tone={value.state === "pending" && inbox ? "warning" : undefined}
            >
              {"priority" in value ? `${value.priority} · ` : ""}
              {stateLabel(value.state)}
            </TimelineBadge>
          );
          if ("event" in value) {
            const source = resourceSource(
              value.event.source,
              state.settings.language,
            );
            return {
              id: value.event.id,
              title: resourceSummary(value.event.description, source),
              date: value.event.event_ts,
              leading: (
                <span className="timeline-card-leading">
                  <FileText size={15} aria-hidden />
                </span>
              ),
              metadata: (
                <>
                  <span>{source}</span>
                  {badge}
                </>
              ),
            };
          }
          return {
            id: value.id,
            title: resourceSummary(value.message, value.from_agent_name),
            date: value.delivered_at ?? value.created_at,
            metadata: (
              <>
                {(!embedded || value.from_agent_name !== agentName) && (
                  <span>{agentLabel(value.from_agent_name)}</span>
                )}
                {badge}
              </>
            ),
            target: { kind: "mail" as const, agentName, id: value.id },
          };
        })}
      >
        {item &&
          ("event" in item ? (
            <article className="detail-article">
              <DetailHeading
                title={eventDetail?.title ?? eventSource}
                metadata={
                  <>
                    {eventDetail?.title !== eventSource && (
                      <span>{eventSource}</span>
                    )}
                    <time dateTime={item.event.event_ts}>
                      {formatDate(item.event.event_ts, state.settings.language)}
                    </time>
                    <TimelineBadge
                      tone={item.state === "pending" ? "warning" : undefined}
                    >
                      {stateLabel(item.state)}
                    </TimelineBadge>
                  </>
                }
              />
              {eventDetail?.content && (
                <p className="detail-prose">{eventDetail.content}</p>
              )}
              {(item.consume_ts ||
                item.event.paths.length > 0 ||
                eventSource !== item.event.source) && (
                <div className="detail-section">
                  {item.consume_ts && (
                    <p className="resource-timestamp">
                      {t("processedAt")}{" "}
                      <time dateTime={item.consume_ts}>
                        {formatDate(item.consume_ts, state.settings.language)}
                      </time>
                    </p>
                  )}
                  {item.event.paths.length > 0 && (
                    <details className="resource-files">
                      <summary>
                        {t("relatedFiles")}{" "}
                        <span className="count-badge">
                          {item.event.paths.length}
                        </span>
                      </summary>
                      <div className="resource-links">
                        {item.event.paths.map((path) => (
                          <button
                            key={path}
                            type="button"
                            className="resource-path"
                            title={path}
                            onClick={() => open(path)}
                          >
                            <span>
                              {path.split(/[\\/]/).at(-1) || path}
                              {path.includes("/") && <small>{path}</small>}
                            </span>
                            <ArrowUpRight size={14} aria-hidden />
                          </button>
                        ))}
                      </div>
                    </details>
                  )}
                  {eventSource !== item.event.source && (
                    <details className="resource-original">
                      <summary>{t("originalInformation")}</summary>
                      <p>{item.event.source}</p>
                    </details>
                  )}
                </div>
              )}
            </article>
          ) : (
            <ResourceMessage
              key={item.id}
              state={state}
              sender={
                embedded && item.from_agent_name === agentName
                  ? ""
                  : agentLabel(item.from_agent_name)
              }
              message={{
                id: item.id,
                role: "assistant",
                source: "proactive",
                content: item.message,
                createdAt: item.delivered_at ?? item.created_at,
                completed: true,
                streaming: false,
                status: null,
              }}
              sourcePath={item.id}
              target={{ kind: "mail", agentName, id: item.id }}
              metadata={
                <TimelineBadge>
                  {item.priority} · {stateLabel(item.state)}
                </TimelineBadge>
              }
            />
          ))}
      </ResourcePane>
    </>
  );
}
