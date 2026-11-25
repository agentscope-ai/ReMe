import { useEffect, useState } from "react";
import type { BackendStatus } from "../../client/src/backend.ts";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import {
  BackendUnlock,
  needsBackendToken,
  useBackendRequest,
} from "./backend-access.tsx";
import { useHost } from "./host.ts";
import BackendSettingsPanel from "./backend-settings.tsx";

/** Keep model-dependent pages behind setup while the management service stays reachable. */
export default function BackendSetup({
  state,
  editableEndpoint,
  onBlockingChange,
}: {
  state: StudioState;
  editableEndpoint: boolean;
  onBlockingChange: (blocking: boolean) => void;
}) {
  const { request, token } = useBackendRequest();
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [status, setStatus] = useState<BackendStatus>();
  const [error, setError] = useState<unknown>(null);
  const [endpoint, setEndpoint] = useState(state.settings.endpoint);
  const [changing, setChanging] = useState(false);
  useEffect(() => {
    setEndpoint(state.settings.endpoint);
    setStatus(undefined);
    setError(null);
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const next = await request<BackendStatus>({
          type: "backend",
          action: "status",
        });
        if (live) {
          setStatus(next);
          setError(null);
        }
      } catch (reason) {
        if (live) setError(reason);
      } finally {
        if (live) timer = setTimeout(() => void poll(), 2000);
      }
    };
    queueMicrotask(() => {
      if (live) void poll();
    });
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [state.settings.endpoint, token]);
  const setup = status?.needs_setup;
  const connecting = !state.clientInfo && !status?.ready;
  useEffect(() => {
    onBlockingChange(Boolean(setup || connecting));
  }, [setup, connecting, onBlockingChange]);
  if (!setup && !connecting) return null;
  return (
    <section
      className="backend-setup"
      aria-label={t(setup ? "backendSetupTitle" : "backendConnectTitle")}
    >
      <div className="backend-setup-inner">
        <span className="backend-brand">ReMe Studio</span>
        <h1>{t(setup ? "backendSetupTitle" : "backendConnectTitle")}</h1>
        <p className="page-description">
          {t(setup ? "backendSetupHint" : "backendConnectHint")}
        </p>
        {setup && (
          <BackendSettingsPanel
            language={state.settings.language}
            active
            setup
          />
        )}
        {!setup && (
          <p role="status" className="settings-note">
            {error
              ? errorText(error, state.settings.language)
              : (status?.error ?? t("backendLoading"))}
          </p>
        )}
        {needsBackendToken(error) && (
          <BackendUnlock language={state.settings.language} />
        )}
        {editableEndpoint && (
          <details open={!setup} className="backend-connection">
            <summary>{t("serviceConnection")}</summary>
            <form
              onSubmit={async (event) => {
                event.preventDefault();
                setChanging(true);
                try {
                  await action({ type: "settings.update", endpoint });
                } catch (reason) {
                  setError(reason);
                } finally {
                  setChanging(false);
                }
              }}
            >
              <label className="backend-field">
                <span>{t("endpoint")}</span>
                <input
                  type="url"
                  aria-label={t("endpoint")}
                  required
                  value={endpoint}
                  onChange={(event) => setEndpoint(event.target.value)}
                />
              </label>
              <button type="submit" className="secondary" disabled={changing}>
                {t("connect")}
              </button>
            </form>
          </details>
        )}
      </div>
    </section>
  );
}
