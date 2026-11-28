import { Mic, Volume2 } from "lucide-react";
import type {
  BackendSettings,
  BackendValues,
} from "../../../client/src/backend.ts";
import { translator, type Language } from "../../../client/src/i18n.ts";
import TextField from "../backend-field.tsx";
import { Switch } from "../ui.tsx";
import "./settings.css";

type SpeechField =
  | "asr_enabled"
  | "tts_enabled"
  | "speech_base_url"
  | "asr_model"
  | "tts_model"
  | "tts_voice";

export default function SpeechServiceSettings({
  language,
  config,
  draft,
  apiKey,
  onKeyChange,
  onChange,
  onReset,
}: {
  language: Language;
  config: BackendSettings;
  draft: BackendValues;
  apiKey: string;
  onKeyChange: (value: string) => void;
  onChange: <K extends SpeechField>(field: K, value: BackendValues[K]) => void;
  onReset: () => void;
}) {
  const t = translator(language);
  const enabled = draft.asr_enabled || draft.tts_enabled;
  const field = (
    name: "asr_model" | "tts_model" | "tts_voice",
    label: "backendAsrModel" | "backendTtsModel" | "backendTtsVoice",
  ) => (
    <TextField
      label={t(label)}
      value={draft[name]}
      required={name === "asr_model" ? draft.asr_enabled : draft.tts_enabled}
      language={language}
      onChange={(value) => onChange(name, value)}
    />
  );
  return (
    <>
      <legend>{t("backendSpeechGroup")}</legend>
      <p className="settings-note">{t("backendSpeechHint")}</p>
      {!config.asr_available && !config.tts_available && (
        <p className="settings-note">{t("speechUnavailable")}</p>
      )}
      <div className="speech-services">
        {(
          [
            [
              "asr",
              "voiceInput",
              "speechRecognitionHint",
              "backendEnableAsr",
              Mic,
            ],
            [
              "tts",
              "voicePlayback",
              "speechSynthesisHint",
              "backendEnableTts",
              Volume2,
            ],
          ] as const
        ).map(
          ([kind, label, hint, action, Icon]) =>
            config[`${kind}_available`] && (
              <section className="speech-service" key={kind}>
                <div className="speech-service-heading">
                  <span className="speech-service-icon">
                    <Icon size={18} aria-hidden />
                  </span>
                  <div>
                    <h3>{t(label)}</h3>
                    <p>{t(hint)}</p>
                  </div>
                  <Switch
                    aria-label={t(action)}
                    checked={draft[`${kind}_enabled`]}
                    onCheckedChange={(value) =>
                      onChange(`${kind}_enabled`, value)
                    }
                  />
                </div>
                <div className="speech-service-fields">
                  {kind === "asr" ? (
                    field("asr_model", "backendAsrModel")
                  ) : (
                    <>
                      {field("tts_model", "backendTtsModel")}
                      {field("tts_voice", "backendTtsVoice")}
                    </>
                  )}
                </div>
              </section>
            ),
        )}
      </div>
      <div className="speech-connection">
        <h3>{t("speechConnection")}</h3>
        <TextField
          label={t("backendDashscopeKey")}
          value={apiKey}
          type="password"
          required={enabled}
          configured={config.dashscope_api_key_set}
          language={language}
          hint={t("backendSpeechKeyHint")}
          onChange={onKeyChange}
        />
        <div className="speech-connection-options">
          <TextField
            label={t("backendSpeechUrl")}
            value={draft.speech_base_url}
            type="url"
            required={enabled}
            language={language}
            hint={t("backendSpeechUrlHint")}
            onChange={(value) => onChange("speech_base_url", value)}
          />
          <button
            type="button"
            className="subtle speech-reset"
            onClick={onReset}
          >
            {t("backendResetSpeech")}
          </button>
        </div>
      </div>
    </>
  );
}
