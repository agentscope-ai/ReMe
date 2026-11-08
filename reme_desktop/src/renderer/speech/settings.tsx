import { useEffect, useState } from "react";
import { translator } from "../../shared/i18n.ts";
import type { DesktopAction, DesktopState } from "../../shared/protocol.ts";

export default function VoiceSettings({
  state,
  run,
}: {
  state: DesktopState;
  run: (action: DesktopAction) => Promise<unknown>;
}) {
  const t = translator(state.settings.language);
  const [volume, setVolume] = useState(state.settings.voiceVolume);
  const [flags, setFlags] = useState({
    voiceAutoSend: state.settings.voiceAutoSend,
    voiceEnabled: state.settings.voiceEnabled,
    voiceReplies: state.settings.voiceReplies,
    voiceReminders: state.settings.voiceReminders,
  });
  const [saving, setSaving] = useState(false);
  useEffect(
    () =>
      setFlags({
        voiceAutoSend: state.settings.voiceAutoSend,
        voiceEnabled: state.settings.voiceEnabled,
        voiceReplies: state.settings.voiceReplies,
        voiceReminders: state.settings.voiceReminders,
      }),
    [
      state.settings.voiceAutoSend,
      state.settings.voiceEnabled,
      state.settings.voiceReplies,
      state.settings.voiceReminders,
    ],
  );
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
        <label className="quiet-row" key={key}>
          <span>{t(key)}</span>
          <input
            type="checkbox"
            aria-label={t(key)}
            checked={flags[key]}
            disabled={
              saving ||
              (key === "voiceAutoSend"
                ? !state.speech.asr
                : !state.speech.tts ||
                  (key !== "voiceEnabled" && !flags.voiceEnabled))
            }
            onChange={async (event) => {
              const checked = event.target.checked;
              setFlags((flags) => ({ ...flags, [key]: checked }));
              setSaving(true);
              const saved = await run({
                type: "settings.update",
                [key]: checked,
              });
              if (saved === false)
                setFlags((flags) => ({
                  ...flags,
                  [key]: state.settings[key],
                }));
              setSaving(false);
            }}
          />
        </label>
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
          onKeyUp={(event) =>
            void run({
              type: "settings.update",
              voiceVolume: Number(event.currentTarget.value),
            })
          }
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
