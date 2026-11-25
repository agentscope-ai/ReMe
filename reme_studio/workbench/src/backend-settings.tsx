import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";
import {
  reasoningEfforts,
  type BackendOperation,
  type BackendPatch,
  type BackendSettings,
  type BackendStatus,
  type BackendValues,
  type NavigationGuard,
  type ReasoningEffort,
} from "../../client/src/backend.ts";
import { errorText, translator, type Language } from "../../client/src/i18n.ts";
import {
  BackendUnlock,
  needsBackendToken,
  useBackendRequest,
} from "./backend-access.tsx";
import { Switch } from "./ui.tsx";
import "./backend-settings.css";

const fields = [
  "workspace_dir",
  "timezone",
  "language",
  "harness",
  "base_url",
  "model",
  "effort",
  "asr_enabled",
  "tts_enabled",
  "speech_base_url",
  "asr_model",
  "tts_model",
  "tts_voice",
] as const;
const speechFields = [
  ["asr_model", "backendAsrModel"],
  ["tts_model", "backendTtsModel"],
  ["tts_voice", "backendTtsVoice"],
] as const;

function TextField({
  label,
  value,
  onChange,
  required = false,
  hint,
  type = "text",
  configured = false,
  language,
}: {
  label: string;
  value: string | null;
  onChange: (value: string) => void;
  required?: boolean;
  hint?: string;
  type?: string;
  configured?: boolean;
  language: Language;
}) {
  const t = translator(language);
  return (
    <label className="backend-field">
      <span>
        {label}
        <span className="backend-field-rule">
          {t(required ? "backendRequired" : "backendOptional")}
        </span>
      </span>
      <input
        type={type}
        value={value ?? ""}
        required={required && !configured}
        autoComplete={type === "password" ? "off" : undefined}
        placeholder={configured ? t("backendKeySet") : undefined}
        onChange={(event) => onChange(event.target.value)}
      />
      {hint && <span className="settings-note">{hint}</span>}
    </label>
  );
}

export default function BackendSettingsPanel({
  language,
  active,
  setup = false,
  registerGuard,
}: {
  language: Language;
  active: boolean;
  setup?: boolean;
  registerGuard?: (guard: NavigationGuard | null) => void;
}) {
  const t = translator(language);
  const { request, token } = useBackendRequest();
  const [config, setConfig] = useState<BackendSettings>();
  const [draft, setDraft] = useState<BackendValues>();
  const [key, setKey] = useState("");
  const [speechKey, setSpeechKey] = useState("");
  const [status, setStatus] = useState<BackendStatus>();
  const [operationId, setOperationId] = useState<string>();
  const [working, setWorking] = useState(false);
  const [tested, setTested] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [navigation, setNavigation] = useState<{ next: () => void } | null>(
    null,
  );
  const navigationRef = useRef(navigation);
  navigationRef.current = navigation;
  const dialog = useRef<HTMLDialogElement>(null);
  const form = useRef<HTMLFormElement>(null);
  const patch: BackendPatch = { revision: config?.revision ?? 0 };
  if (draft && config) {
    for (const field of fields) {
      if (draft[field] !== config[field])
        Object.assign(patch, { [field]: draft[field] });
    }
    if (key.trim()) patch.api_key = key.trim();
    if (speechKey.trim()) patch.dashscope_api_key = speechKey.trim();
  }
  const dirty = Object.keys(patch).length > 1;
  const restarting =
    status?.operation?.phase === "accepted" ||
    status?.operation?.phase === "restarting";
  const busy = working || Boolean(operationId) || restarting;
  const stale = Boolean(
    config && status && config.revision !== status.revision,
  );
  const speechEnabled = Boolean(draft?.asr_enabled || draft?.tts_enabled);
  const complete = Boolean(
    draft?.workspace_dir?.trim() &&
    (!config?.model_editable ||
      (draft.harness &&
        draft.base_url?.trim() &&
        draft.model?.trim() &&
        (key.trim() || config.api_key_set))) &&
    (!speechEnabled ||
      ((speechKey.trim() || config?.dashscope_api_key_set) &&
        draft?.speech_base_url?.trim())) &&
    (!draft?.asr_enabled || draft.asr_model.trim()) &&
    (!draft?.tts_enabled || (draft.tts_model.trim() && draft.tts_voice.trim())),
  );

  const adopt = (saved: BackendSettings) => {
    setConfig(saved);
    setDraft(saved);
    setKey("");
    setSpeechKey("");
    setTested(false);
  };
  useEffect(() => {
    if (!active) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await request<BackendStatus>({
          type: "backend",
          action: "status",
          operationId,
        });
        if (!live) return;
        setStatus(next);
        if (
          (operationId && next.operation?.phase === "succeeded") ||
          (!operationId && !dirty && next.revision !== config?.revision)
        ) {
          const saved = await request<BackendSettings>({
            type: "backend",
            action: "config",
          });
          if (!live) return;
          adopt(saved);
          setOperationId(undefined);
          if (operationId) {
            const pending = navigationRef.current;
            setNavigation(null);
            pending?.next();
          }
        }
        if (operationId && next.operation?.phase === "failed")
          setOperationId(undefined);
        setLoadError(null);
      } catch (reason) {
        if (live) setLoadError(reason);
      } finally {
        if (live) timer = setTimeout(() => void poll(), 1200);
      }
    };
    void poll();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [active, operationId, config?.revision, dirty, token]);
  useEffect(() => {
    registerGuard?.(dirty ? (next) => setNavigation({ next }) : null);
    const unload = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault();
    };
    window.addEventListener("beforeunload", unload);
    return () => {
      registerGuard?.(null);
      window.removeEventListener("beforeunload", unload);
    };
  }, [dirty, registerGuard]);
  useEffect(() => {
    if (navigation) dialog.current?.showModal();
    else dialog.current?.close();
  }, [navigation]);
  useEffect(() => {
    if (setup && config) {
      form.current?.querySelector<HTMLInputElement>("input:invalid")?.focus();
    }
  }, [setup, config?.revision]);

  const submit = async (action: "apply" | "restart" | "test_model") => {
    setWorking(true);
    setError(null);
    setTested(false);
    try {
      const settings =
        action === "test_model"
          ? {
              revision: patch.revision,
              workspace_dir: draft?.workspace_dir,
              harness: draft?.harness,
              base_url: draft?.base_url,
              model: draft?.model,
              effort: draft?.effort,
              ...(key.trim() ? { api_key: key.trim() } : {}),
            }
          : patch;
      const operation = await request<BackendOperation>(
        action === "restart"
          ? { type: "backend", action }
          : { type: "backend", action, settings },
      );
      if (action === "test_model") setTested(true);
      else setOperationId(operation.id);
    } catch (reason) {
      setError(reason);
    } finally {
      setWorking(false);
    }
  };
  const update = <K extends keyof BackendValues>(
    field: K,
    value: BackendValues[K],
  ) => {
    setDraft((previous) =>
      previous ? { ...previous, [field]: value } : previous,
    );
    setTested(false);
  };
  const input = (
    field:
      | "workspace_dir"
      | "base_url"
      | "model"
      | "speech_base_url"
      | "asr_model"
      | "tts_model"
      | "tts_voice",
    label: string,
    required: boolean,
    hint?: string,
    type?: string,
  ) => (
    <TextField
      label={label}
      value={draft?.[field] ?? ""}
      required={required}
      language={language}
      hint={hint}
      type={type}
      onChange={(value) => update(field, value)}
    />
  );
  const issue = error ?? loadError ?? config?.settings_error;
  return (
    <div className="backend-settings">
      {issue != null && (
        <p className="action-error" role="alert">
          {errorText(issue, language)}
        </p>
      )}
      {needsBackendToken(loadError) ? (
        <BackendUnlock language={language} />
      ) : !config || !draft ? (
        <p role="status">
          {t(loadError ? "backendOffline" : "backendLoading")}
        </p>
      ) : (
        <form
          ref={form}
          onSubmit={(event) => {
            event.preventDefault();
            void submit("apply");
          }}
        >
          <p className="backend-status settings-note" role="status">
            {t(
              busy
                ? "backendRestarting"
                : status?.ready
                  ? "backendReady"
                  : status?.needs_setup
                    ? "backendSetupNeeded"
                    : "backendOffline",
            )}
          </p>
          {status?.error && status.operation?.phase !== "failed" && (
            <p className="action-error" role="alert">
              {status.error}
            </p>
          )}
          {status?.operation?.phase === "failed" && (
            <p className="action-error" role="alert">
              {status.operation.error}
            </p>
          )}
          {stale && dirty && (
            <p className="settings-note">
              {t("backendSettingsChanged")}{" "}
              <button
                type="button"
                className="subtle"
                disabled={busy}
                onClick={async () => {
                  try {
                    adopt(
                      await request<BackendSettings>({
                        type: "backend",
                        action: "config",
                      }),
                    );
                  } catch (reason) {
                    setError(reason);
                  }
                }}
              >
                {t("backendReload")}
              </button>
            </p>
          )}
          <fieldset disabled={busy} className="backend-section">
            <legend>{t("backendWorkspaceGroup")}</legend>
            {input(
              "workspace_dir",
              t("backendWorkspace"),
              true,
              t("backendWorkspaceHint"),
            )}
            {!setup && (
              <details className="backend-advanced">
                <summary>
                  <span>{t("backendAdvanced")}</span>
                  <ChevronDown size={16} aria-hidden />
                </summary>
                <div className="backend-model-fields">
                  <TextField
                    label={t("backendTimezone")}
                    value={draft.timezone}
                    language={language}
                    onChange={(value) => update("timezone", value || null)}
                    hint={t("backendTimezoneHint")}
                  />
                  <label className="backend-field">
                    <span>
                      {t("backendLanguage")}
                      <span className="backend-field-rule">
                        {t("backendOptional")}
                      </span>
                    </span>
                    <select
                      value={draft.language}
                      onChange={(event) =>
                        update("language", event.target.value)
                      }
                    >
                      <option value="">{t("backendDefault")}</option>
                      <option value="zh">简体中文</option>
                      <option value="en">English</option>
                    </select>
                  </label>
                </div>
              </details>
            )}
          </fieldset>
          {config.model_editable && (
            <fieldset disabled={busy} className="backend-section">
              <legend>{t("backendHarnessGroup")}</legend>
              <label className="backend-field">
                <span>
                  {t("backendHarness")}
                  <span className="backend-field-rule">
                    {t("backendRequired")}
                  </span>
                </span>
                <select
                  value={draft.harness ?? "codex"}
                  onChange={() => update("harness", "codex")}
                >
                  <option value="codex">Codex</option>
                </select>
                <span className="settings-note">{t("backendHarnessHint")}</span>
              </label>
              {input(
                "base_url",
                t("backendUrl"),
                true,
                t("backendUrlHint"),
                "url",
              )}
              <TextField
                label={t("backendKey")}
                value={key}
                type="password"
                required
                configured={config.api_key_set}
                language={language}
                hint={
                  config.api_key_set ? t("backendKeySet") : t("backendKeyHint")
                }
                onChange={(value) => {
                  setKey(value);
                  setTested(false);
                }}
              />
              <div className="backend-model-fields">
                {input("model", t("backendModel"), true)}
                <label className="backend-field">
                  <span>
                    {t("backendEffort")}
                    <span className="backend-field-rule">
                      {t("backendOptional")}
                    </span>
                  </span>
                  <select
                    value={draft.effort ?? ""}
                    onChange={(event) =>
                      update(
                        "effort",
                        (event.target.value || null) as ReasoningEffort | null,
                      )
                    }
                  >
                    <option value="">{t("backendDefault")}</option>
                    {reasoningEfforts.map((effort) => (
                      <option key={effort}>{effort}</option>
                    ))}
                  </select>
                </label>
              </div>
            </fieldset>
          )}
          {(!setup || speechEnabled) && config.dashscope_editable && (
            <fieldset disabled={busy} className="backend-section">
              <legend>
                {t("backendSpeechGroup")}
                <span className="backend-field-rule">
                  {t("backendOptional")}
                </span>
              </legend>
              <p className="settings-note">{t("backendSpeechHint")}</p>
              <div className="backend-toggles">
                {(["asr", "tts"] as const).map(
                  (kind) =>
                    config[`${kind}_available`] && (
                      <div className="backend-toggle-row" key={kind}>
                        <span>
                          {t(
                            kind === "asr"
                              ? "backendEnableAsr"
                              : "backendEnableTts",
                          )}
                        </span>
                        <Switch
                          aria-label={t(
                            kind === "asr"
                              ? "backendEnableAsr"
                              : "backendEnableTts",
                          )}
                          checked={draft[`${kind}_enabled`]}
                          onCheckedChange={(value) =>
                            update(`${kind}_enabled`, value)
                          }
                        />
                      </div>
                    ),
                )}
              </div>
              {(speechEnabled ||
                (!config.asr_available && !config.tts_available)) && (
                <>
                  <TextField
                    label={t("backendDashscopeKey")}
                    value={speechKey}
                    type="password"
                    required={speechEnabled}
                    configured={config.dashscope_api_key_set}
                    language={language}
                    hint={
                      config.dashscope_api_key_set
                        ? t("backendKeySet")
                        : t("backendSpeechKeyHint")
                    }
                    onChange={setSpeechKey}
                  />
                  {speechEnabled && (
                    <>
                      <button
                        type="button"
                        className="subtle backend-reset"
                        onClick={() =>
                          setDraft((previous) =>
                            previous
                              ? { ...previous, ...config.defaults }
                              : previous,
                          )
                        }
                      >
                        {t("backendResetSpeech")}
                      </button>
                      {input(
                        "speech_base_url",
                        t("backendSpeechUrl"),
                        true,
                        t("backendSpeechUrlHint"),
                        "url",
                      )}
                      <div className="backend-model-fields">
                        {speechFields.map(
                          ([field, label]) =>
                            (field === "asr_model"
                              ? draft.asr_enabled
                              : draft.tts_enabled) && (
                              <span key={field}>
                                {input(field, t(label), true)}
                              </span>
                            ),
                        )}
                      </div>
                    </>
                  )}
                </>
              )}
            </fieldset>
          )}
          {tested && (
            <p role="status" className="settings-note">
              {t("backendTestPassed")}
            </p>
          )}
          {patch.workspace_dir !== undefined && !setup && (
            <p className="settings-note">{t("backendWorkspaceChange")}</p>
          )}
          <p className="settings-note">{t("backendStorageHint")}</p>
          <footer>
            {config.model_editable && (
              <button
                type="button"
                className="subtle"
                disabled={
                  busy ||
                  !draft.base_url?.trim() ||
                  !draft.model?.trim() ||
                  !(key.trim() || config.api_key_set)
                }
                onClick={() => void submit("test_model")}
              >
                {t("backendTest")}
              </button>
            )}
            {!setup && (
              <button
                type="button"
                className="secondary"
                disabled={busy || !complete}
                onClick={() => void submit("restart")}
              >
                {t("backendRestart")}
              </button>
            )}
            <button
              type="submit"
              className="primary"
              disabled={
                busy || stale || !complete || (!dirty && config.initialized)
              }
            >
              {t(setup ? "backendFinishSetup" : "backendApply")}
            </button>
          </footer>
        </form>
      )}
      <dialog
        ref={dialog}
        className="backend-confirm"
        onCancel={() => setNavigation(null)}
      >
        <h3>{t("backendUnsaved")}</h3>
        <p>{t("backendUnsavedHint")}</p>
        <footer>
          <button
            type="button"
            className="subtle"
            onClick={() => setNavigation(null)}
          >
            {t("backendKeepEditing")}
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => {
              const pending = navigation;
              setNavigation(null);
              pending?.next();
            }}
          >
            {t("backendDiscard")}
          </button>
          <button
            type="button"
            className="primary"
            disabled={!complete || busy || stale}
            onClick={() => {
              dialog.current?.close();
              form.current?.requestSubmit();
            }}
          >
            {t("backendApply")}
          </button>
        </footer>
      </dialog>
    </div>
  );
}
