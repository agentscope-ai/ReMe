import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { errorText, translator } from "../../../client/src/i18n.ts";
import type { StudioState } from "../../../client/src/protocol.ts";
import type { TaskDocument, TaskSettings } from "../../../client/src/tasks.ts";
import { formatDate, useResource } from "../resources.ts";
import TaskSettingsFields from "./settings.tsx";
import {
  changedTaskSettings,
  taskSettings,
} from "../../../client/src/tasks.ts";
import "./tasks.css";

export type TaskTarget =
  | { mode: "create"; withSchedule: boolean }
  | { mode: "edit" | "delete"; agentName: string };

export default function TaskDialog({
  target,
  state,
  onClose,
  onCreated,
  onSaved,
}: {
  target: TaskTarget;
  state: StudioState;
  onClose: () => void;
  onCreated: (agentName: string) => void;
  onSaved: () => void;
}) {
  const requestResource = useResource();
  const language = state.settings.language;
  const t = translator(language);
  const create = target.mode === "create";
  const agentName = create ? "" : target.agentName;
  const agent = state.agents?.find((value) => value.agent_name === agentName);
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [document, setDocument] = useState<TaskDocument>();
  const [baseline, setBaseline] = useState(() => agent && taskSettings(agent));
  const [settings, setSettings] = useState<TaskSettings>(
    () =>
      baseline ?? {
        enabled: true,
        cron: create && target.withSchedule ? "0 8 * * *" : null,
        min_trigger_interval_seconds: 0,
        notify_window: null,
      },
  );
  const [tab, setTab] = useState<"description" | "settings">("description");
  const [loading, setLoading] = useState(target.mode === "edit");
  const [retry, setRetry] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState(false);
  const modal = useRef<HTMLDialogElement>(null);
  const live = useRef(false);
  const afterClose = useRef<() => void>(onClose);
  const close = (complete = onClose) => {
    if (!modal.current?.open) return;
    afterClose.current = complete;
    modal.current.close();
  };
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
    if (target.mode !== "edit") return;
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
  }, [agentName, retry, target.mode]);
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
        if (live.current) close(() => onCreated(result.agent_name));
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
        const changes = changedTaskSettings(baseline!, settings);
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
    });
  const remove = () =>
    perform(async () => {
      await requestResource({
        type: "task",
        action: "delete",
        agent_name: agentName,
      });
      if (live.current) close();
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
        if (!busy) close();
      }}
      onClose={(event) => {
        const dialog = event.currentTarget;
        if (dialog.open) return;
        const animations = dialog.getAnimations();
        void Promise.allSettled(
          animations.map((animation) => animation.finished),
        ).then(() => {
          if (live.current && !dialog.open) afterClose.current();
        });
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
          onClick={() => close()}
        >
          <X size={18} />
        </button>
      </header>
      {!create && target.mode !== "delete" && (
        <nav className="task-dialog-tabs" aria-label={t("taskManage")}>
          {(["description", "settings"] as const).map((id) => (
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
              {t(id === "description" ? "taskTab" : "taskSettings")}
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
            onClick={() => {
              setError(null);
              setSaved(false);
              if (agent) {
                const value = taskSettings(agent);
                setBaseline(value);
                setSettings(value);
              }
              setRetry((value) => value + 1);
            }}
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
              onClick={() => close()}
            >
              {t("taskCancel")}
            </button>
            <button
              type="button"
              className="task-danger"
              disabled={busy || !agent}
              onClick={() => void remove()}
            >
              {t("taskDelete")}
            </button>
          </footer>
        </div>
      ) : (
        <form
          className="task-dialog-body"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <fieldset disabled={busy || (!create && !agent)}>
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
                key={retry}
                value={settings}
                timezone={state.clientInfo?.timezone ?? null}
                language={language}
                onChange={(value) => {
                  setSettings(value);
                  setSaved(false);
                }}
              />
            )}
            {agent && tab === "settings" && (
              <details className="task-advanced">
                <summary>{t("taskDetails")}</summary>
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
                {agent.error && <p className="action-error">{agent.error}</p>}
              </details>
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
                  : !agent ||
                    !baseline ||
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
