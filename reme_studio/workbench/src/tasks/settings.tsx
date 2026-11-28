import { useId } from "react";
import { Plus, Trash2 } from "lucide-react";
import { type Language, translator } from "../../../client/src/i18n.ts";
import { Switch } from "../ui.tsx";
import {
  changeSchedulePreset,
  scheduleAtTime,
  defaultTaskTrigger,
  defaultNotifyWindows,
  parseScheduleTime,
  type ScheduleDraft,
  type SchedulePreset,
  type TaskSettingsDraft,
} from "./settings-draft.ts";

type Translate = ReturnType<typeof translator>;

function ScheduleFields({
  value,
  onChange,
  timezone,
  t,
}: {
  value: ScheduleDraft;
  onChange: (schedule: ScheduleDraft) => void;
  timezone: string | null;
  t: Translate;
}) {
  const id = useId();
  const { preset, schedule } = value;
  return (
    <div className="task-field">
      <label htmlFor={id}>{t("taskSchedule")}</label>
      <div className="task-field-row">
        <select
          id={id}
          value={preset}
          onChange={(event) =>
            onChange(
              changeSchedulePreset(value, event.target.value as SchedulePreset),
            )
          }
        >
          <option value="daily">{t("taskDaily")}</option>
          <option value="weekdays">{t("taskWeekdays")}</option>
          <option value="hourly">{t("taskHourly")}</option>
          <option value="custom">{t("taskCustomSchedule")}</option>
        </select>
        {(preset === "daily" || preset === "weekdays") && (
          <input
            type="time"
            aria-label={t("taskTime")}
            required
            value={parseScheduleTime(schedule)?.time ?? ""}
            onChange={(event) =>
              onChange({
                preset,
                schedule: scheduleAtTime(preset, event.target.value),
              })
            }
          />
        )}
      </div>
      {preset === "custom" && (
        <input
          aria-label={t("taskScheduleExpression")}
          value={schedule}
          required
          placeholder="0 8 * * *"
          onChange={(event) =>
            onChange({ preset, schedule: event.target.value })
          }
        />
      )}
      <small>
        {t("taskTimezone", { timezone: timezone ?? t("taskSystemTimezone") })}
      </small>
    </div>
  );
}

function EventFields({
  minIntervalSeconds,
  onChange,
  t,
}: {
  minIntervalSeconds: number;
  onChange: (seconds: number) => void;
  t: Translate;
}) {
  return (
    <>
      <p className="task-hint">{t("taskEventsHint")}</p>
      <label className="task-field">
        <span>{t("taskMinInterval")}</span>
        <input
          type="number"
          min="0"
          step="any"
          required
          value={minIntervalSeconds}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </label>
    </>
  );
}

export default function TaskSettingsFields({
  create,
  value,
  onChange,
  timezone,
  language,
}: {
  create: boolean;
  value: TaskSettingsDraft;
  onChange: (value: TaskSettingsDraft) => void;
  timezone: string | null;
  language: Language;
}) {
  const t = translator(language);
  const triggerId = useId();
  const { trigger, enabled, notify_windows: notifyWindows } = value;
  const patch = (settings: Partial<TaskSettingsDraft>) =>
    onChange({ ...value, ...settings });
  return (
    <>
      <div className="task-field">
        <label htmlFor={triggerId}>{t("taskTriggers")}</label>
        {create ? (
          <select
            id={triggerId}
            value={trigger.type}
            onChange={(event) =>
              patch({
                trigger: defaultTaskTrigger(
                  event.target.value === "time" ? "time" : "event",
                ),
              })
            }
          >
            <option value="time">{t("taskTimeTrigger")}</option>
            <option value="event">{t("taskEventTrigger")}</option>
          </select>
        ) : (
          <input
            id={triggerId}
            readOnly
            value={t(
              trigger.type === "time" ? "taskTimeTrigger" : "taskEventTrigger",
            )}
          />
        )}
      </div>
      {trigger.type === "time" ? (
        <ScheduleFields
          value={trigger.schedule}
          onChange={(schedule) =>
            patch({ trigger: { type: "time", schedule } })
          }
          timezone={timezone}
          t={t}
        />
      ) : (
        <EventFields
          minIntervalSeconds={trigger.minIntervalSeconds}
          onChange={(minIntervalSeconds) =>
            patch({ trigger: { type: "event", minIntervalSeconds } })
          }
          t={t}
        />
      )}
      <label className="task-switch">
        <span>{t("taskEnabled")}</span>
        <Switch
          checked={enabled}
          onCheckedChange={(enabled) => patch({ enabled })}
        />
      </label>
      <label className="task-switch">
        <span>{t("taskLimitNotifications")}</span>
        <Switch
          checked={notifyWindows.length > 0}
          onCheckedChange={(limited) =>
            patch({
              notify_windows: limited ? defaultNotifyWindows() : [],
            })
          }
        />
      </label>
      {notifyWindows.length > 0 && (
        <div className="task-notify-windows">
          {notifyWindows.map((window, index) => (
            <div className="task-notify-window" key={index}>
              {(["start", "end"] as const).map((key) => (
                <label className="task-field" key={key}>
                  <span>
                    {t(key === "start" ? "taskWindowStart" : "taskWindowEnd")}
                  </span>
                  <input
                    type="time"
                    required
                    value={window[key]}
                    onChange={(event) =>
                      patch({
                        notify_windows: notifyWindows.map((item, position) =>
                          position === index
                            ? { ...item, [key]: event.target.value }
                            : item,
                        ),
                      })
                    }
                  />
                </label>
              ))}
              <button
                type="button"
                className="icon-button"
                aria-label={t("taskRemoveWindow", { index: index + 1 })}
                title={t("taskRemoveWindow", { index: index + 1 })}
                onClick={() =>
                  patch({
                    notify_windows: notifyWindows.filter(
                      (_, position) => position !== index,
                    ),
                  })
                }
              >
                <Trash2 size={16} aria-hidden />
              </button>
            </div>
          ))}
          <button
            type="button"
            className="task-add-window"
            onClick={() =>
              patch({
                notify_windows: [...notifyWindows, { start: "", end: "" }],
              })
            }
          >
            <Plus size={16} aria-hidden />
            {t("taskAddWindow")}
          </button>
        </div>
      )}
    </>
  );
}
