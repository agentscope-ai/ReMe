import { useEffect, useRef, useState } from "react";
import { errorText, translator } from "../../../client/src/i18n.ts";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import type {
  TaskDocument,
  ProactiveRuntimeSettings,
} from "../../../client/src/tasks.ts";
import { formatDate, useResource } from "../resources.ts";
import TaskSettingsFields from "./settings.tsx";
import {
  changedTaskSettings,
  taskSettings,
} from "../../../client/src/tasks.ts";
import "./tasks.css";

const tabs = [
  ["description", "taskTab"],
  ["settings", "taskSettings"],
  ["details", "taskDetails"],
] as const;

export default function TaskEditor({
  agent,
  state,
  onCreated,
}: {
  agent?: AgentView;
  state: StudioState;
  onCreated: (agentName: string) => void;
}) {
  const requestResource = useResource();
  const language = state.settings.language;
  const t = translator(language);
  const agentName = agent?.agent_name;
  const create = !agentName;
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [document, setDocument] = useState<TaskDocument>();
  const [baseline, setBaseline] = useState(() => agent && taskSettings(agent));
  const [settings, setSettings] = useState<ProactiveRuntimeSettings>(
    () =>
      baseline ?? {
        enabled: true,
        cron: null,
        min_trigger_interval_seconds: 0,
        notify_window: null,
      },
  );
  const [tab, setTab] = useState<(typeof tabs)[number][0]>("description");
  const [loading, setLoading] = useState(!create);
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);
  const live = useRef(false);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  useEffect(() => {
    if (!agentName) return;
    let current = true;
    setLoading(true);
    setError(null);
    void requestResource<TaskDocument>({
      type: "task",
      action: "read",
      agent_name: agentName,
    })
      .then((text) => {
        if (current) {
          setDocument(text);
          setContent(text.content);
        }
      })
      .catch((reason) => {
        if (current) setError(reason);
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [agentName, retry]);
  const save = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      if (!agentName) {
        const result = await requestResource<{ agent_name: string }>({
          type: "task",
          action: "create",
          agent_name: name,
          task_description: content,
          settings,
        });
        if (live.current) onCreated(result.agent_name);
      } else if (tab === "description" && document) {
        const result = await requestResource<
          Pick<TaskDocument, "mtime" | "path">
        >({
          type: "task",
          action: "save",
          agent_name: agentName,
          content,
          expected_mtime: document.mtime,
        });
        if (live.current) {
          setDocument({ ...document, ...result, content });
          setSaved(true);
        }
      } else if (tab === "settings" && baseline) {
        const changes = changedTaskSettings(baseline, settings);
        if (Object.keys(changes).length)
          await requestResource({
            type: "task",
            action: "update",
            agent_name: agentName,
            settings: changes,
          });
        if (live.current) {
          setBaseline(settings);
          setSaved(true);
        }
      }
    } catch (reason) {
      if (live.current) setError(reason);
    } finally {
      if (live.current) setBusy(false);
    }
  };
  return (
    <div className="task-editor">
      {agent && (
        <nav className="task-editor-tabs" aria-label={t("taskManage")}>
          {tabs.map(([id, label]) => (
            <button
              type="button"
              key={id}
              aria-current={tab === id ? "page" : undefined}
              disabled={busy}
              onClick={() => {
                setTab(id);
                setSaved(false);
              }}
            >
              {t(label)}
            </button>
          ))}
        </nav>
      )}
      {tab !== "details" && error != null && (
        <div className="task-editor-error" role="alert">
          {errorText(error, language)}
          {agent && (
            <button
              type="button"
              className="subtle"
              disabled={busy}
              onClick={() => {
                setError(null);
                setSaved(false);
                const value = taskSettings(agent);
                setBaseline(value);
                setSettings(value);
                setRetry((value) => value + 1);
              }}
            >
              {t("taskReload")}
            </button>
          )}
        </div>
      )}
      {tab === "details" && agent ? (
        <div>
          <dl className="task-facts">
            <dt>{t("taskLastStarted")}</dt>
            <dd>{formatDate(agent.last_started_at, language)}</dd>
            <dt>{t("taskLastCompleted")}</dt>
            <dd>{formatDate(agent.last_completed_at, language)}</dd>
            <dt>{t("nextTrigger")}</dt>
            <dd>{formatDate(agent.next_trigger_at, language)}</dd>
            <dt>{t("pendingEvents")}</dt>
            <dd>{agent.pending_events}</dd>
            <dt>{t("taskInFlight")}</dt>
            <dd>{agent.in_flight_events}</dd>
          </dl>
          {agent.error && (
            <p className="action-error" role="alert">
              {agent.error}
            </p>
          )}
        </div>
      ) : loading ? (
        <p className="task-hint">{t("documentLoading")}</p>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset disabled={busy}>
            {create && (
              <label className="task-field">
                <span>{t("taskName")}</span>
                <input
                  autoFocus
                  required
                  maxLength={80}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="tech_news"
                />
              </label>
            )}
            {(create || tab === "description") && (
              <label className="task-field">
                <span>{t("taskTab")}</span>
                <textarea
                  required
                  rows={9}
                  maxLength={100000}
                  value={content}
                  placeholder={t("taskDescriptionPlaceholder")}
                  onChange={(event) => {
                    setContent(event.target.value);
                    setSaved(false);
                  }}
                />
              </label>
            )}
            {(create || tab === "settings") && (
              <TaskSettingsFields
                key={retry}
                create={create}
                value={settings}
                timezone={state.clientInfo?.timezone ?? null}
                language={language}
                onChange={(value) => {
                  setSettings(value);
                  setSaved(false);
                }}
              />
            )}
            {agent?.status === "active" && (
              <p className="task-hint">
                {t(agent.enabled ? "taskChangesHint" : "taskPauseHint")}
              </p>
            )}
          </fieldset>
          <footer>
            <span role="status">{saved ? t("taskSaved") : ""}</span>
            <button
              className="primary"
              disabled={
                busy ||
                (create
                  ? !name.trim() || !content.trim()
                  : !baseline ||
                    (tab === "description" && (!document || !content.trim())))
              }
            >
              {t(busy ? "taskSaving" : create ? "taskCreate" : "taskSave")}
            </button>
          </footer>
        </form>
      )}
    </div>
  );
}
