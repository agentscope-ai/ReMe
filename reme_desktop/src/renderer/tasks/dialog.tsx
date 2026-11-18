import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { errorText, translator } from "../../shared/i18n.ts";
import type { DesktopState, WorkspaceConfig } from "../../shared/protocol.ts";
import type {
  TaskDocument,
  TaskSettings,
  TaskStatus,
  TaskType,
} from "../../shared/tasks.ts";
import { formatDate, requestResource } from "../resources.ts";
import TaskSettingsFields from "./settings.tsx";
import { changedTaskSettings, taskSettings } from "../../shared/tasks.ts";
import "./tasks.css";

export type TaskTarget =
  | { mode: "create"; agentType: TaskType }
  | { mode: "edit" | "status" | "delete"; agentName: string };

export default function TaskDialog({
  target,
  state,
  onClose,
  onCreated,
  onSaved,
}: {
  target: TaskTarget;
  state: DesktopState;
  onClose: () => void;
  onCreated: (agentName: string) => void;
  onSaved: () => void;
}) {
  const language = state.settings.language;
  const t = translator(language);
  const create = target.mode === "create";
  const agentName = create ? "" : target.agentName;
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [document, setDocument] = useState<TaskDocument>();
  const [status, setStatus] = useState<TaskStatus>();
  const [settings, setSettings] = useState<TaskSettings>({
    agent_type: create ? target.agentType : "cron",
    enabled: true,
    cron: create && target.agentType === "proactive" ? null : "0 8 * * *",
    min_trigger_interval_seconds: 0,
    notify_window: null,
  });
  const [tab, setTab] = useState(
    target.mode === "status" ? "status" : "description",
  );
  const [timezone, setTimezone] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);
  const modal = useRef<HTMLDialogElement>(null);
  const live = useRef(false);
  useEffect(() => {
    live.current = true;
    const element = modal.current!;
    element.showModal();
    return () => {
      live.current = false;
      element.close();
    };
  }, []);
  useEffect(() => {
    let current = true;
    setLoading(true);
    setError(null);
    const load = async () => {
      const [config, value] = await Promise.all([
        requestResource<WorkspaceConfig>({ type: "config" }),
        create
          ? undefined
          : requestResource<TaskStatus>({
              type: "task",
              action: "status",
              agent_name: agentName,
            }),
      ]);
      if (!current) return;
      setTimezone(config.timezone);
      setStatus(value);
      if (value) setSettings(taskSettings(value));
      if (!create && target.mode !== "delete") {
        const text = await requestResource<TaskDocument>({
          type: "task",
          action: "read",
          agent_name: agentName,
        });
        if (current) {
          setDocument(text);
          setContent(text.content);
        }
      }
    };
    void load()
      .catch((reason) => {
        if (current) setError(reason);
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [agentName, create, retry, target.mode]);
  const latest = state.agents?.find((agent) => agent.agent_name === agentName);
  const facts = status ? { ...status, ...latest } : undefined;
  const perform = async (operation: () => Promise<void>) => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSaved(false);
    try {
      await operation();
    } catch (reason) {
      if (live.current) setError(reason);
    } finally {
      if (live.current) setBusy(false);
    }
  };
  const save = () =>
    perform(async () => {
      if (create) {
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
          onSaved();
        }
      } else {
        const changes = changedTaskSettings(taskSettings(status!), settings);
        if (Object.keys(changes).length)
          await requestResource({
            type: "task",
            action: "update",
            agent_name: agentName,
            settings: changes,
          });
        if (live.current) {
          setStatus({
            ...status!,
            ...settings,
            cron: settings.cron ?? undefined,
          });
          setSaved(true);
        }
      }
    });
  const remove = () =>
    perform(async () => {
      await requestResource({
        type: "task",
        action: "delete",
        agent_name: agentName,
      });
      if (live.current) onClose();
    });
  const title = t(
    create
      ? "taskCreate"
      : target.mode === "delete"
        ? "taskDelete"
        : "taskManage",
  );
  return (
    <dialog
      ref={modal}
      className="task-dialog"
      aria-labelledby="task-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onClose();
      }}
    >
      <header>
        <div>
          <h2 id="task-dialog-title">{title}</h2>
          {!create && <p>{agentName}</p>}
        </div>
        <button
          type="button"
          className="icon-button"
          aria-label={t("taskClose")}
          disabled={busy}
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      {!create && target.mode !== "delete" && (
        <nav className="task-dialog-tabs" aria-label={t("taskManage")}>
          {(["description", "settings", "status"] as const).map((id) => (
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
              {t(
                id === "description"
                  ? "taskTab"
                  : id === "settings"
                    ? "taskSettings"
                    : "taskDetails",
              )}
            </button>
          ))}
        </nav>
      )}
      {error != null && (
        <div className="task-dialog-error" role="alert">
          {errorText(error, language)}
          <button
            type="button"
            className="subtle"
            disabled={busy}
            onClick={() => setRetry((value) => value + 1)}
          >
            {t("taskReload")}
          </button>
        </div>
      )}
      {loading ? (
        <p className="task-dialog-loading">{t("documentLoading")}</p>
      ) : target.mode === "delete" ? (
        <div className="task-dialog-body">
          <p>{t("taskDeleteHint", { name: agentName })}</p>
          <footer>
            <button
              type="button"
              className="secondary"
              disabled={busy}
              onClick={onClose}
            >
              {t("taskCancel")}
            </button>
            <button
              type="button"
              className="task-danger"
              disabled={busy || !status}
              onClick={() => void remove()}
            >
              {t("taskDelete")}
            </button>
          </footer>
        </div>
      ) : !create && tab === "status" ? (
        <div className="task-dialog-body">
          {facts && (
            <dl className="task-facts">
              <dt>{t("taskState")}</dt>
              <dd>
                {t(
                  !facts.enabled
                    ? "agentDisabled"
                    : facts.error
                      ? "agentError"
                      : facts.running
                        ? "agentRunning"
                        : "agentIdle",
                )}
              </dd>
              <dt>{t("taskLastStarted")}</dt>
              <dd>{formatDate(facts.last_started_at, language)}</dd>
              <dt>{t("taskLastCompleted")}</dt>
              <dd>{formatDate(facts.last_completed_at, language)}</dd>
              <dt>{t("nextTrigger")}</dt>
              <dd>{formatDate(facts.next_trigger_at, language)}</dd>
              <dt>{t("pendingEvents")}</dt>
              <dd>{facts.pending_events}</dd>
              <dt>{t("taskInFlight")}</dt>
              <dd>{facts.in_flight_events}</dd>
            </dl>
          )}
          {facts?.error && <p className="action-error">{facts.error}</p>}
          {facts?.running && !facts.enabled && (
            <p className="task-hint">{t("taskPauseHint")}</p>
          )}
        </div>
      ) : (
        <form
          className="task-dialog-body"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset disabled={busy || (!create && !status)}>
            {create && (
              <div className="task-field-row">
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
                <div className="task-field">
                  <label htmlFor="task-type">{t("taskType")}</label>
                  <select
                    id="task-type"
                    value={settings.agent_type}
                    onChange={(event) => {
                      const agent_type = event.target.value as TaskType;
                      setSettings({
                        ...settings,
                        agent_type,
                        cron: agent_type === "cron" ? "0 8 * * *" : null,
                        min_trigger_interval_seconds: 0,
                      });
                    }}
                  >
                    <option value="cron">{t("scheduledTasks")}</option>
                    <option value="proactive">{t("proactiveTasks")}</option>
                  </select>
                </div>
              </div>
            )}
            {(create || tab === "description") && (
              <div className="task-field">
                <label htmlFor="task-description">{t("taskTab")}</label>
                <textarea
                  id="task-description"
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
              </div>
            )}
            {(create || tab === "settings") && (
              <TaskSettingsFields
                key={settings.agent_type}
                value={settings}
                timezone={timezone}
                language={language}
                onChange={(value) => {
                  setSettings(value);
                  setSaved(false);
                }}
              />
            )}
            {facts?.running && (
              <p className="task-hint">{t("taskChangesHint")}</p>
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
                  : !status ||
                    (tab === "description" && (!document || !content.trim())))
              }
            >
              {t(busy ? "taskSaving" : create ? "taskCreate" : "taskSave")}
            </button>
          </footer>
        </form>
      )}
    </dialog>
  );
}
