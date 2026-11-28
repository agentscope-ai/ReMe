import { useContext, useEffect, useId, useRef, useState } from "react";
import { translator } from "../../../client/src/i18n.ts";
import type {
  StudioAction,
  StudioState,
} from "../../../client/src/protocol.ts";
import { Switch } from "../ui.tsx";
import { WorkspaceLayout } from "../workspace-titlebar.tsx";
import "./settings.css";

export default function VoiceSettings({
  state,
  run,
  disabled = false,
}: {
  state: StudioState;
  run: (action: StudioAction) => Promise<unknown>;
  disabled?: boolean;
}) {
  const t = translator(state.settings.language);
  const mobile = useContext(WorkspaceLayout)?.mobile;
  const [volume, setVolume] = useState(state.settings.voiceVolume);
  const [saving, setSaving] = useState(false);
  const volumeId = useId();
  const keyboardCommit = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  useEffect(() => () => clearTimeout(keyboardCommit.current), []);
  useEffect(
    () => setVolume(state.settings.voiceVolume),
    [state.settings.voiceVolume],
  );
  const toggle = (
    key: "voiceAutoSend" | "voiceEnabled" | "voiceReplies" | "voiceReminders",
  ) => (
    <div className="speech-preference-row" key={key}>
      <span>{t(key)}</span>
      <Switch
        aria-label={t(key)}
        checked={state.settings[key]}
        disabled={
          disabled ||
          saving ||
          (key === "voiceAutoSend"
            ? !state.speech.asr
            : !state.speech.tts ||
              (key !== "voiceEnabled" && !state.settings.voiceEnabled))
        }
        onCheckedChange={async (checked) => {
          setSaving(true);
          try {
            await run({ type: "settings.update", [key]: checked });
          } finally {
            setSaving(false);
          }
        }}
      />
    </div>
  );
  const saveVolume = (value: number) => {
    if (!disabled && state.speech.tts && value !== state.settings.voiceVolume)
      void run({ type: "settings.update", voiceVolume: value });
  };
  return (
    <section className="voice-preferences" aria-label={t("speechPreferences")}>
      <p className="speech-preferences-note">{t("voicePreferencesHint")}</p>
      <section className="speech-preference-group">
        <h4>{t("speechSendBehavior")}</h4>
        {toggle("voiceAutoSend")}
        <p className="settings-note">
          {t(
            state.speech.asr
              ? mobile
                ? "speechMobileAutoSendHint"
                : "speechAutoSendHint"
              : "speechNeedsRecognition",
          )}
        </p>
        <div className="speech-recording-help">
          <h5>{t("speechRecordingHelp")}</h5>
          <p className="settings-note">
            {t(mobile ? "voiceInputMobileHint" : "voiceInputHint")}
          </p>
          {!mobile && <p className="settings-note">{t("voiceShortcut")}</p>}
        </div>
      </section>
      <section className="speech-preference-group">
        <h4>{t("speechPlaybackBehavior")}</h4>
        {toggle("voiceEnabled")}
        {!state.speech.tts && (
          <p className="settings-note">{t("speechNeedsSynthesis")}</p>
        )}
        <div className="speech-auto-options">
          {(["voiceReplies", "voiceReminders"] as const).map(toggle)}
        </div>
        {!state.settings.voiceEnabled && (
          <p className="settings-note">{t("speechManualPlaybackHint")}</p>
        )}
        <div className="speech-volume">
          <label htmlFor={volumeId}>{t("voiceVolume")}</label>
          <output htmlFor={volumeId}>{Math.round(volume * 100)}%</output>
          <input
            id={volumeId}
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={volume}
            disabled={disabled || saving || !state.speech.tts}
            onChange={(event) => setVolume(Number(event.target.value))}
            onPointerUp={(event) =>
              saveVolume(Number(event.currentTarget.value))
            }
            onKeyUp={(event) => {
              if (
                ![
                  "ArrowLeft",
                  "ArrowRight",
                  "ArrowUp",
                  "ArrowDown",
                  "Home",
                  "End",
                  "PageUp",
                  "PageDown",
                ].includes(event.key)
              )
                return;
              const value = Number(event.currentTarget.value);
              clearTimeout(keyboardCommit.current);
              keyboardCommit.current = setTimeout(() => saveVolume(value), 150);
            }}
          />
        </div>
        <p className="settings-note">{t("voiceHint")}</p>
      </section>
    </section>
  );
}
