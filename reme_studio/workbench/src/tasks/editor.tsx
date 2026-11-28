import { useEffect, useRef, useState } from "react";
import { errorText, translator } from "../../../client/src/i18n.ts";
import type { AgentView, StudioState } from "../../../client/src/protocol.ts";
import {
  changedTaskSettings,
  taskSettings,
} from "../../../client/src/tasks.ts";
import { useResource } from "../resources.ts";
import { taskSettingsDraft, toRuntimeSettings } from "./settings-draft.ts";
import TaskSettingsFields from "./settings.tsx";
import "./tasks.css";

export default function TaskEditor({
  agent,
  state,
  onCreated,
  initialTrigger = "event",
}: {
  agent?: AgentView;
  state: StudioState;
  onCreated: (agentName: string) => void;
  initialTrigger?: "event" | "time";
}) {
  const requestResource = useResource();
  const language = state.settings.language;
  const t = translator(language);
  const agentName = agent?.agent_name;
  const create = !agentName;
  const [name, setName] = useState("");
  const [content, setContent] = useState("");
  const [baseline, setBaseline] = useState(() => agent && taskSettings(agent));
  const [settings, setSettings] = useState(() =>
    taskSettingsDraft(
      baseline,
      agent ? (agent.agent_type === "time" ? "time" : "event") : initialTrigger,
    ),
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [saved, setSaved] = useState<"settings" | "create" | null>(null);
  const live = useRef(false);
  useEffect(() => {
    live.current = true;
    return () => {
      live.current = false;
    };
  }, []);
  const save = async (section: "settings" | "create") => {
    if (busy) return;
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      if (!agentName) {
        const result = await requestResource<{ agent_name: string }>({
          type: "task",
          action: "create",
          agent_type: settings.trigger.type,
          agent_name: name,
          task_description: content,
          settings: toRuntimeSettings(settings),
        });
        if (live.current) onCreated(result.agent_name);
      } else if (section === "settings" && baseline) {
        const updated = toRuntimeSettings(settings, baseline);
        const changes = changedTaskSettings(baseline, updated);
        if (Object.keys(changes).length)
          await requestResource({
            type: "task",
            action: "update",
            agent_name: agentName,
            settings: changes,
          });
        if (live.current) {
          setBaseline(updated);
          setSaved(section);
        }
      }
    } catch (reason) {
      if (live.current) setError(reason);
    } finally {
      if (live.current) setBusy(false);
    }
  };
  const description = (
    <label className="task-field">
      <span>{t("taskTab")}</span>
      <textarea
        required
        rows={7}
        maxLength={100000}
        value={content}
        placeholder={t("taskDescriptionPlaceholder")}
        onChange={(event) => {
          setContent(event.target.value);
          setSaved(null);
        }}
      />
    </label>
  );
  const fields = (
    <TaskSettingsFields
      create={create}
      value={settings}
      timezone={state.clientInfo?.timezone ?? null}
      language={language}
      onChange={(value) => {
        setSettings(value);
        setSaved(null);
      }}
    />
  );
  const footer = (section: "settings" | "create") => (
    <footer>
      <span role="status">{saved === section ? t("taskSaved") : ""}</span>
      <button
        className="primary"
        disabled={
          busy ||
          (section === "create" ? !name.trim() || !content.trim() : !baseline)
        }
      >
        {t(
          busy
            ? "taskSaving"
            : section === "create"
              ? "taskCreate"
              : "flowSaveSettings",
        )}
      </button>
    </footer>
  );
  return (
    <div className="task-editor">
      {error != null && (
        <div className="task-editor-error" role="alert">
          {errorText(error, language)}
          {agent && (
            <button
              type="button"
              className="subtle"
              disabled={busy}
              onClick={() => {
                setError(null);
                setSaved(null);
                const value = taskSettings(agent);
                setBaseline(value);
                setSettings(
                  taskSettingsDraft(
                    value,
                    agent.agent_type === "time" ? "time" : "event",
                  ),
                );
              }}
            >
              {t("taskReload")}
            </button>
          )}
        </div>
      )}
      {create ? (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void save("create");
          }}
        >
          <fieldset disabled={busy}>
            <label className="task-field">
              <span>{t("taskName")}</span>
              <input
                autoFocus
                required
                maxLength={80}
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder="proactive_news"
              />
            </label>
            {description}
            {fields}
          </fieldset>
          {footer("create")}
        </form>
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void save("settings");
          }}
        >
          <fieldset disabled={busy}>{fields}</fieldset>
          {footer("settings")}
        </form>
      )}
      {agent?.status === "active" && (
        <p className="task-hint">
          {t(agent.enabled ? "taskChangesHint" : "taskPauseHint")}
        </p>
      )}
    </div>
  );
}
