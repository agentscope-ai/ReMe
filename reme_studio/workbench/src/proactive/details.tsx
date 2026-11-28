import { useEffect, useState, type ReactNode } from "react";
import { ArrowUpRight, Radio } from "lucide-react";
import type { StudioState } from "../../../client/src/protocol.ts";
import type {
  OverviewAgent,
  OverviewInboxItem,
  OverviewReminder,
  SourceRef,
} from "../../../client/src/wire.ts";
import { errorText, translator } from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import { formatDate, useResource } from "../resources.ts";
import MarkdownText from "../markdown.tsx";
import { splitReminderSource } from "../message-body.tsx";
import { sourceIcon } from "./flow.ts";

type Fact = [string, ReactNode];
type Linked = (paths: string[]) => ReactNode;
type SourceLabel = (source: SourceRef) => string;
const present = ([, value]: Fact) =>
  value != null && value !== "" && value !== false;
const date = (value: string | null | undefined, state: StudioState) =>
  value ? formatDate(value, state.settings.language) : null;

export function Facts({ values }: { values: Fact[] }) {
  const items = values.filter(present);
  return items.length ? (
    <dl className="flow-facts">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  ) : null;
}

export function TraceDetails({
  values,
  label,
}: {
  values: Fact[];
  label: string;
}) {
  return values.some(present) ? (
    <details className="flow-technical">
      <summary>{label}</summary>
      <Facts values={values} />
    </details>
  ) : null;
}

function Sources({
  sources,
  label,
  showType = true,
}: {
  sources: SourceRef[];
  label: SourceLabel;
  showType?: boolean;
}) {
  const unique = [
    ...new Map(sources.map((source) => [source.source_id, source])).values(),
  ];
  return unique.length ? (
    <div className="flow-detail-sources">
      {unique.map((source) => {
        const Icon = sourceIcon(source);
        return (
          <span key={source.source_id}>
            <Icon size={14} strokeWidth={1.6} aria-hidden />
            {label(source)}
            {showType && (
              <small
                className="flow-type"
                data-time={source.trigger_type === "time"}
              >
                {source.trigger_type}
              </small>
            )}
          </span>
        );
      })}
    </div>
  ) : null;
}

export function AgentDetail({
  agent,
  state,
  status,
  linked,
  sourceLabel,
  onOutbox,
}: {
  agent: OverviewAgent;
  state: StudioState;
  status: string;
  linked: Linked;
  sourceLabel: SourceLabel;
  onOutbox: () => void;
}) {
  const t = translator(state.settings.language);
  const run = agent.active_run ?? agent.last_run;
  const errors = [...new Set([agent.error, run?.error].filter(Boolean))];
  return (
    <>
      <div className="flow-status-card">
        <span className="flow-detail-status">
          <Radio size={16} />
          {status}
        </span>
        <span className="flow-type" data-time={agent.agent_type === "time"}>
          {agent.agent_type}
        </span>
      </div>
      <Facts
        values={[
          [
            t("flowNextAt"),
            agent.agent_type === "time" && date(agent.next_trigger_at, state),
          ],
          [t("flowRetryAt"), date(agent.retry_at, state)],
          [
            t("flowWaitingUntil"),
            agent.waiting_until !== agent.retry_at &&
              date(agent.waiting_until, state),
          ],
          [t("flowTool"), agent.active_run?.tool],
        ]}
      />
      {errors.map((error) => (
        <p className="action-error" role="alert" key={error}>
          {error}
        </p>
      ))}
      {run && (
        <section className="flow-detail-section">
          <div className="flow-section-heading">
            <h3>{t(agent.active_run ? "flowRun" : "flowLastRun")}</h3>
            <span>{date(run.completed_at ?? run.started_at, state)}</span>
          </div>
          <Sources
            sources={run.source_refs}
            label={sourceLabel}
            showType={false}
          />
          <p className="flow-run-summary">
            <span>{t("flowInputCount", { count: run.event_ids.length })}</span>
            {run.phase === "completed" && <span>{t("completed")}</span>}
          </p>
          {run.phase === "completed" &&
            run.notification_requested === false && (
              <p className="flow-hint">{t("flowNoReminder")}</p>
            )}
        </section>
      )}
      {Boolean(run?.record_paths.length) && (
        <section className="flow-detail-section">
          <h3>{t("flowOutputs")}</h3>
          {linked(run!.record_paths)}
        </section>
      )}
      {agent.pending_delivery_count > 0 && (
        <button className="flow-pending-link" onClick={onOutbox}>
          {t("flowDelivering", { count: agent.pending_delivery_count })}
          <ArrowUpRight size={14} />
        </button>
      )}
      <TraceDetails
        label={t("originalInformation")}
        values={[
          [t("taskName"), agent.agent_name],
          [t("flowRun"), run?.run_id],
          [t("flowInputs"), [...new Set(run?.event_ids)].join(" · ")],
        ]}
      />
    </>
  );
}

export function EventDetail({
  state,
  item,
  linked,
  sourceLabel,
  onInbox,
}: {
  state: StudioState;
  item: OverviewInboxItem;
  linked: Linked;
  sourceLabel: SourceLabel;
  onInbox: () => void;
}) {
  const t = translator(state.settings.language);
  const request = useResource();
  const { action } = useHost();
  const [message, setMessage] = useState<string>();
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    if (!item.notification) return;
    let live = true;
    setMessage(undefined);
    setError(null);
    void request<{ message: string }>({
      type: "reminder",
      agentName: "Remy",
      id: item.notification.record_path,
    })
      .then((value) => {
        if (live) setMessage(value.message);
      })
      .catch((reason) => {
        if (live) setError(reason);
      });
    return () => {
      live = false;
    };
  }, [item.id]);
  const body = item.event?.description ?? message;
  const files = [
    ...new Set([
      ...(item.event?.paths ?? []),
      ...item.record_paths,
      ...(item.notification ? [item.notification.record_path] : []),
    ]),
  ];
  return (
    <>
      <div className="flow-detail-meta">
        <span className="flow-state-pill" data-state={item.state}>
          {t(
            item.state === "pending"
              ? "flowPending"
              : item.state === "processing"
                ? "flowProcessing"
                : "flowHandled",
          )}
        </span>
        <span>{date(item.received_at, state)}</span>
        <span>
          {item.notification?.source_agent ?? item.agent_name}
          {item.notification && " → Remy"}
        </span>
      </div>
      <Sources sources={item.source_refs} label={sourceLabel} />
      <div className="flow-message-body">
        {body ? (
          <MarkdownText
            text={body}
            onLink={(url) =>
              void action({ type: "link.open", url }).catch(setError)
            }
          />
        ) : (
          <p className="flow-hint">
            {t(error ? "flowRecordReady" : "documentLoading")}
          </p>
        )}
      </div>
      {error != null && (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      )}
      <Facts
        values={[
          [
            t("flowOccurredAt"),
            date(item.event?.event_ts, state) !==
              date(item.received_at, state) &&
              date(item.event?.event_ts, state),
          ],
          [
            t("processedAt"),
            date(item.consume_ts, state) !== date(item.received_at, state) &&
              date(item.consume_ts, state),
          ],
        ]}
      />
      {files.length > 0 && (
        <section className="flow-detail-section">
          <h3>{t("relatedFiles")}</h3>
          {linked(files)}
        </section>
      )}
      <button className="flow-text-link" onClick={onInbox}>
        {t("flowOpenInbox")}
        <ArrowUpRight size={14} />
      </button>
      <TraceDetails
        label={t("originalInformation")}
        values={[
          [t("flowReceivedAt"), item.received_at],
          [t("flowOccurredAt"), item.event?.event_ts],
          [t("processedAt"), item.consume_ts],
          [t("flowInputs"), [...new Set(item.event_ids)].join(" · ")],
          [t("flowRun"), [...new Set(item.run_ids)].join(" · ")],
        ]}
      />
    </>
  );
}

export function ReminderDetail({
  state,
  item,
  linked,
  sourceLabel,
}: {
  state: StudioState;
  item: OverviewReminder;
  linked: Linked;
  sourceLabel: SourceLabel;
}) {
  const request = useResource();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [content, setContent] = useState(item.summary);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    let live = true;
    setContent(item.summary);
    setError(null);
    void request<{ message?: { content: string } }>({
      type: "reminder",
      agentName: "Remy",
      id: item.id,
    })
      .then((value) => {
        if (live && value.message) setContent(value.message.content);
      })
      .catch((reason) => {
        if (live) setError(reason);
      });
    return () => {
      live = false;
    };
  }, [item.id]);
  const source = splitReminderSource(
    {
      id: item.id,
      role: "assistant",
      source: "proactive",
      content,
      createdAt: item.delivered_at,
      completed: true,
      streaming: false,
      status: null,
    },
    state.clientInfo?.workspace_dir,
  );
  const files = [
    ...new Set([...item.record_paths, ...(source ? [source.path] : [])]),
  ];
  return (
    <>
      <div className="flow-detail-meta">
        <span className="flow-state-pill" data-state="processed">
          {t("flowSent")}
        </span>
        <span>{date(item.delivered_at, state)}</span>
      </div>
      <Sources sources={item.source_refs} label={sourceLabel} />
      <div className="flow-message-body">
        <MarkdownText
          text={source?.display.content ?? content}
          onLink={(url) =>
            void action({ type: "link.open", url }).catch(setError)
          }
        />
      </div>
      {error != null && (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      )}
      {files.length > 0 && (
        <section className="flow-detail-section">
          <h3>{t("relatedFiles")}</h3>
          {linked(files)}
        </section>
      )}
      <TraceDetails
        label={t("originalInformation")}
        values={[
          [t("flowInputs"), [...new Set(item.event_ids)].join(" · ")],
          [t("flowRun"), [...new Set(item.run_ids)].join(" · ")],
        ]}
      />
    </>
  );
}
