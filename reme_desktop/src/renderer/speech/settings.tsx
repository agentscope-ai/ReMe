import { useEffect, useRef, useState } from "react";
import { translator } from "../../shared/i18n.ts";
import type { DesktopAction, DesktopState } from "../../shared/protocol.ts";
import { Switch } from "../ui.tsx";

export default function VoiceSettings({
  state,
  run,
}: {
  state: DesktopState;
  run: (action: DesktopAction) => Promise<unknown>;
}) {
  const t = translator(state.settings.language);
  const [volume, setVolume] = useState(state.settings.voiceVolume);
  const [saving, setSaving] = useState(false);
  const keyboardCommit = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  useEffect(() => () => clearTimeout(keyboardCommit.current), []);
  useEffect(
    () => setVolume(state.settings.voiceVolume),
    [state.settings.voiceVolume],
  );
  return (
    <fieldset className="voice-settings">
      <legend>{t("speechSettings")}</legend>
      {!state.speech.tts && (
        <p className="settings-note">{t("speechTtsUnavailable")}</p>
      )}
      {(
        [
          "voiceAutoSend",
          "voiceEnabled",
          "voiceReplies",
          "voiceReminders",
        ] as const
      ).map((key) => (
        <div className="quiet-row" key={key}>
          <span>{t(key)}</span>
          <Switch
            aria-label={t(key)}
            checked={state.settings[key]}
            disabled={
              saving ||
              (key === "voiceAutoSend"
                ? !state.speech.asr
                : !state.speech.tts ||
                  (key !== "voiceEnabled" && !state.settings.voiceEnabled))
            }
            onCheckedChange={async (checked) => {
              setSaving(true);
              await run({
                type: "settings.update",
                [key]: checked,
              });
              setSaving(false);
            }}
          />
        </div>
      ))}
      <label className="scale-row">
        {t("voiceVolume")} <span>{Math.round(volume * 100)}%</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          aria-label={t("voiceVolume")}
          value={volume}
          onChange={(event) => setVolume(Number(event.target.value))}
          onPointerUp={(event) =>
            void run({
              type: "settings.update",
              voiceVolume: Number(event.currentTarget.value),
            })
          }
          onKeyUp={(event) => {
            const voiceVolume = Number(event.currentTarget.value);
            clearTimeout(keyboardCommit.current);
            keyboardCommit.current = setTimeout(
              () => void run({ type: "settings.update", voiceVolume }),
              150,
            );
          }}
        />
      </label>
      <p className="settings-note">{t("voiceHint")}</p>
      <p className="settings-note">
        {state.speech.asr ? t("voiceInputHint") : t("speechAsrUnavailable")}
      </p>
      <p className="settings-note">{t("voiceShortcut")}</p>
    </fieldset>
  );
}
