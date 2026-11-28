import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowUpRight, LoaderCircle } from "lucide-react";
import type { ConnectorView, OverviewAgent } from "../../../client/src/wire.ts";
import type { StudioState } from "../../../client/src/protocol.ts";
import type { ConnectorRequest } from "../../../client/src/connectors.ts";
import { translator, errorText } from "../../../client/src/i18n.ts";
import { formatDate, useResource } from "../resources.ts";
import { Facts, TraceDetails } from "./details.tsx";
import { parseScheduleTime } from "../tasks/settings-draft.ts";

export default function SourceEditor({
  state,
  source,
  agents,
  icon,
  onSave,
  onTask,
  onCreateTime,
}: {
  state: StudioState;
  source?: ConnectorView;
  agents: OverviewAgent[];
  icon?: string;
  onSave: (id: string, icon: string) => void;
  onTask: (name: string) => void;
  onCreateTime: () => void;
}) {
  const t = translator(state.settings.language);
  const request = useResource();
  const [kind, setKind] = useState<"event" | "time">(
    source?.trigger_type ?? "event",
  );
  const [name, setName] = useState(source?.name ?? "");
  const [url, setUrl] = useState("");
  const [emoji, setEmoji] = useState(icon ?? "");
  const [targets, setTargets] = useState(source?.runtime_names ?? []);
  const [enabled, setEnabled] = useState(source?.enabled ?? true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const live = useRef(true);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  const save = async (event: FormEvent) => {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      let id = source?.source_id;
      if (!source) {
        const value = await request<ConnectorView>({
          type: "connector",
          action: "create",
          name,
          runtime_names: targets,
          binding: { url },
        });
        id = value.source_id;
      } else if (kind !== "time") {
        const patch: Extract<ConnectorRequest, { action: "update" }> = {
          type: "connector",
          action: "update",
          source_id: source.source_id,
        };
        if (name.trim() !== source.name) patch.name = name.trim();
        if (enabled !== (source.enabled ?? true)) patch.enabled = enabled;
        if (JSON.stringify(targets) !== JSON.stringify(source.runtime_names))
          patch.runtime_names = targets;
        if (Object.keys(patch).length > 3) await request(patch);
      }
      if (id && live.current) onSave(id, emoji.trim());
    } catch (reason) {
      if (live.current) setError(reason);
    } finally {
      if (live.current) setBusy(false);
    }
  };
  const remoteChanged =
    !source ||
    (kind !== "time" &&
      (name.trim() !== source.name ||
        enabled !== (source.enabled ?? true) ||
        JSON.stringify(targets) !== JSON.stringify(source.runtime_names)));
  const parsedSchedule = source?.schedule
    ? parseScheduleTime(source.schedule)
    : undefined;
  const scheduleLabel = parsedSchedule
    ? `${t(parsedSchedule.preset === "daily" ? "taskDaily" : "taskWeekdays")} ${parsedSchedule.time}`
    : source?.schedule === "0 * * * *"
      ? t("taskHourly")
      : t("taskCustomSchedule");
  const initialIcon = icon ?? "";
  return (
    <form className="flow-source-form" onSubmit={(event) => void save(event)}>
      {source?.available === false && (
        <p className="flow-hint">{t("flowUnavailableSource")}</p>
      )}
      {source && (
        <>
          <div className="flow-detail-meta">
            <span className="flow-type" data-time={kind === "time"}>
              {kind}
            </span>
          </div>
          <Facts
            values={[
              [
                t("flowNextAt"),
                source.next_trigger_at &&
                  formatDate(source.next_trigger_at, state.settings.language),
              ],
              [
                t("flowDelivery"),
                (source.pending_delivery_count ?? 0) > 0 &&
                  t("flowDelivering", {
                    count: source.pending_delivery_count!,
                  }),
              ],
            ]}
          />
        </>
      )}
      {source?.last_error && (
        <p className="action-error" role="alert">
          {source.last_error}
        </p>
      )}
      {!source && (
        <div
          className="flow-source-types"
          role="group"
          aria-label={t("flowSourceType")}
        >
          {(["event", "time"] as const).map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={kind === value}
              onClick={() => setKind(value)}
            >
              {t(value === "event" ? "flowEventType" : "flowTimeType")}
            </button>
          ))}
        </div>
      )}
      {kind === "time" ? (
        <>
          {!source && <p className="flow-hint">{t("flowTimeHint")}</p>}
          {source && (
            <>
              {source.schedule && (
                <div className="flow-time-plan">
                  <strong>{scheduleLabel}</strong>
                  <small>{state.clientInfo?.timezone}</small>
                </div>
              )}
              <button
                type="button"
                className="secondary"
                onClick={() => onTask(source.runtime_names[0])}
              >
                {t("flowEditTime")}
                <ArrowUpRight size={14} />
              </button>
            </>
          )}
          {!source && (
            <>
              {agents
                .filter((agent) => agent.agent_type === "time")
                .map((agent) => (
                  <button
                    className="flow-file"
                    key={agent.agent_name}
                    type="button"
                    onClick={() => onTask(agent.agent_name)}
                  >
                    <span>⏰ {agent.agent_name}</span>
                    <ArrowUpRight size={14} />
                  </button>
                ))}
              <button type="button" className="primary" onClick={onCreateTime}>
                {t("flowCreateTime")}
              </button>
            </>
          )}
        </>
      ) : (
        <>
          {!source && <p className="flow-hint">{t("flowSourceHint")}</p>}
          <label>
            {t("flowSourceName")}
            <input
              value={name}
              maxLength={80}
              required
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          {!source && (
            <label>
              {t("flowSourceUrl")}
              <input
                type="url"
                placeholder="https://example.com/events"
                value={url}
                required
                onChange={(event) => setUrl(event.target.value)}
              />
            </label>
          )}
          {source && (
            <label className="flow-check">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(event) => setEnabled(event.target.checked)}
              />
              {t("flowSourceEnabled")}
            </label>
          )}
          <fieldset>
            <legend>{t("flowSourceTargets")}</legend>
            {agents
              .filter((agent) => agent.agent_type === "event")
              .map((agent) => (
                <label className="flow-check" key={agent.agent_name}>
                  <input
                    type="checkbox"
                    checked={targets.includes(agent.agent_name)}
                    onChange={(event) =>
                      setTargets((value) =>
                        event.target.checked
                          ? [...value, agent.agent_name]
                          : value.filter((name) => name !== agent.agent_name),
                      )
                    }
                  />
                  {agent.agent_name}
                </label>
              ))}
            {source && !targets.length && (
              <p className="flow-hint">{t("flowTargetHint")}</p>
            )}
          </fieldset>
        </>
      )}
      {(source || kind === "event") && (
        <>
          <label>
            <span title={t("flowIconHint")}>{t("flowIcon")}</span>
            <input
              className="flow-emoji-input"
              value={emoji}
              maxLength={16}
              onChange={(event) => setEmoji(event.target.value)}
            />
          </label>
          {source && (
            <TraceDetails
              label={t("originalInformation")}
              values={[
                [t("flowSourceId"), source.source_id],
                [t("flowSourceUrl"), source.binding?.url],
                [t("taskScheduleExpression"), source.schedule],
                [
                  t("flowCollectedAt"),
                  source.last_collected_at &&
                    formatDate(
                      source.last_collected_at,
                      state.settings.language,
                    ),
                ],
                [
                  t("flowAdmittedAt"),
                  source.last_admitted_at &&
                    formatDate(
                      source.last_admitted_at,
                      state.settings.language,
                    ),
                ],
              ]}
            />
          )}
          {error != null && (
            <p className="action-error" role="alert">
              {errorText(error, state.settings.language)}
            </p>
          )}
          <button
            className="primary"
            disabled={
              busy ||
              (Boolean(source) &&
                !remoteChanged &&
                emoji.trim() === initialIcon) ||
              (remoteChanged && !state.proactive?.live) ||
              (!source && (!targets.length || !name.trim() || !url))
            }
          >
            {busy && <LoaderCircle size={14} className="flow-spin" />}
            {t(source ? "taskSave" : "flowAddSource")}
          </button>
        </>
      )}
    </form>
  );
}
