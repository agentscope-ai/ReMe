import type { Language } from "../../../client/src/i18n.ts";
import { useState } from "react";
import { translator } from "../../../client/src/i18n.ts";
import type { ProactiveRuntimeSettings } from "../../../client/src/tasks.ts";
import { Switch } from "../ui.tsx";

export default function TaskSettingsFields({
  value,
  onChange,
  timezone,
  language,
}: {
  value: ProactiveRuntimeSettings;
  onChange: (value: ProactiveRuntimeSettings) => void;
  timezone: string | null;
  language: Language;
}) {
  const t = translator(language);
  const patch = (settings: Partial<ProactiveRuntimeSettings>) =>
    onChange({ ...value, ...settings });
  const parts = value.cron?.split(/\s+/) ?? [];
  const [kind, setKind] = useState(() =>
    /^\d+ \d+ \* \* (\*|1-5)$/.test(value.cron ?? "")
      ? parts[4] === "*"
        ? "daily"
        : "weekdays"
      : value.cron === "0 * * * *"
        ? "hourly"
        : "custom",
  );
  const daily = kind === "daily" || kind === "weekdays";
  return (
    <>
      <label className="task-field">
        <span>{t("taskTriggers")}</span>
        <select
          value={value.cron === null ? "events" : "scheduled"}
          onChange={(event) => {
            setKind("daily");
            patch({
              cron: event.target.value === "scheduled" ? "0 8 * * *" : null,
            });
          }}
        >
          <option value="events">{t("taskObserve")}</option>
          <option value="scheduled">{t("taskScheduled")}</option>
        </select>
      </label>
      {value.cron !== null && (
        <div className="task-field">
          <label htmlFor="task-schedule">{t("taskSchedule")}</label>
          <div className="task-field-row">
            <select
              id="task-schedule"
              value={kind}
              onChange={(event) => {
                setKind(event.target.value);
                patch({
                  cron:
                    {
                      daily: "0 8 * * *",
                      weekdays: "0 8 * * 1-5",
                      hourly: "0 * * * *",
                      custom: value.cron,
                    }[event.target.value] ?? "",
                });
              }}
            >
              <option value="daily">{t("taskDaily")}</option>
              <option value="weekdays">{t("taskWeekdays")}</option>
              <option value="hourly">{t("taskHourly")}</option>
              <option value="custom">{t("taskCustomSchedule")}</option>
            </select>
            {daily && (
              <input
                type="time"
                aria-label={t("taskTime")}
                required
                value={`${parts[1].padStart(2, "0")}:${parts[0].padStart(2, "0")}`}
                onChange={(event) => {
                  const [hour, minute] = event.target.value.split(":");
                  if (hour && minute)
                    patch({
                      cron: `${Number(minute)} ${Number(hour)} * * ${parts[4]}`,
                    });
                }}
              />
            )}
          </div>
          {kind === "custom" && (
            <input
              aria-label={t("taskCron")}
              value={value.cron ?? ""}
              required
              placeholder="0 8 * * *"
              onChange={(event) => patch({ cron: event.target.value })}
            />
          )}
          <small>
            {t("taskTimezone", {
              timezone: timezone ?? t("taskSystemTimezone"),
            })}
          </small>
        </div>
      )}
      {value.cron === null && (
        <p className="task-hint">{t("taskEventsHint")}</p>
      )}
      <label className="task-switch">
        <span>{t("taskEnabled")}</span>
        <Switch
          checked={value.enabled}
          onCheckedChange={(enabled) => patch({ enabled })}
        />
      </label>
      <details className="task-advanced">
        <summary>{t("taskAdvanced")}</summary>
        <label className="task-field">
          <span>{t("taskMinInterval")}</span>
          <input
            type="number"
            min="0"
            step="any"
            required
            value={value.min_trigger_interval_seconds}
            onChange={(event) =>
              patch({
                min_trigger_interval_seconds: Number(event.target.value),
              })
            }
          />
        </label>
        <label className="task-switch">
          <span>{t("taskLimitNotifications")}</span>
          <Switch
            checked={value.notify_window !== null}
            onCheckedChange={(limited) =>
              patch({
                notify_window: limited
                  ? { start: "09:00", end: "18:00" }
                  : null,
              })
            }
          />
        </label>
        {value.notify_window && (
          <div className="task-field-row">
            {(["start", "end"] as const).map((key) => (
              <label className="task-field" key={key}>
                <span>
                  {t(key === "start" ? "taskWindowStart" : "taskWindowEnd")}
                </span>
                <input
                  type="time"
                  required
                  value={value.notify_window![key]}
                  onChange={(event) =>
                    patch({
                      notify_window: {
                        ...value.notify_window!,
                        [key]: event.target.value,
                      },
                    })
                  }
                />
              </label>
            ))}
          </div>
        )}
      </details>
    </>
  );
}
