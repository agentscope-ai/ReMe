import { useEffect, useState } from "react";
import {
  ArrowRight,
  Cable,
  Check,
  ChevronDown,
  Circle,
  FolderOpen,
  MessageCircle,
  Sparkles,
} from "lucide-react";
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
import studioIcon from "../public/reme-app-icon.png";
import "./backend-setup.css";

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
    <section className="backend-setup" aria-labelledby="backend-welcome-title">
      <div className="backend-setup-inner">
        <header className="backend-setup-header">
          <span className="backend-brand">
            <img src={studioIcon} width="28" height="28" alt="" />
            ReMe Studio
          </span>
          <select
            aria-label={t("language")}
            value={state.settings.language}
            onChange={(event) =>
              void action({
                type: "settings.update",
                language: event.target.value === "zh" ? "zh" : "en",
              }).catch(setError)
            }
          >
            <option value="zh">简体中文</option>
            <option value="en">English</option>
          </select>
        </header>
        <div className="backend-setup-layout">
          <aside className="backend-welcome">
            <span className="backend-welcome-eyebrow">MEET REMY</span>
            <h1 id="backend-welcome-title">{t("backendWelcomeTitle")}</h1>
            <p className="backend-welcome-hint">{t("backendWelcomeHint")}</p>
            <ul className="backend-welcome-features">
              {(
                [
                  [
                    MessageCircle,
                    "backendWelcomeChat",
                    "backendWelcomeChatHint",
                  ],
                  [
                    FolderOpen,
                    "backendWelcomeMemory",
                    "backendWelcomeMemoryHint",
                  ],
                  [
                    Sparkles,
                    "backendWelcomeProactive",
                    "backendWelcomeProactiveHint",
                  ],
                ] as const
              ).map(([Icon, label, hint]) => (
                <li key={label}>
                  <span className="backend-feature-icon">
                    <Icon size={19} aria-hidden />
                  </span>
                  <div>
                    <h2>{t(label)}</h2>
                    <p>{t(hint)}</p>
                  </div>
                </li>
              ))}
            </ul>
          </aside>
          <main className="backend-setup-content">
            <div className="backend-setup-heading">
              <h2>{t(setup ? "backendSetupTitle" : "backendConnectTitle")}</h2>
              <p>{t(setup ? "backendSetupHint" : "backendConnectHint")}</p>
            </div>
            <details
              open={!setup}
              className="backend-connection backend-setup-connection"
            >
              <summary>
                <Cable size={18} aria-hidden />
                <span className="backend-connection-label">
                  <strong>{t("connectionSettings")}</strong>
                  <span title={state.settings.endpoint}>
                    {state.settings.endpoint}
                  </span>
                </span>
                <span
                  className="backend-connection-state"
                  data-connected={Boolean(status)}
                >
                  {status ? (
                    <Check size={13} aria-hidden />
                  ) : (
                    <Circle size={10} aria-hidden />
                  )}
                  {t(
                    status
                      ? "connected"
                      : error
                        ? "backendNotConnected"
                        : "reconnecting",
                  )}
                </span>
                {editableEndpoint && (
                  <ChevronDown
                    className="backend-connection-chevron"
                    size={15}
                    aria-hidden
                  />
                )}
              </summary>
              {editableEndpoint && (
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
                      placeholder="http://127.0.0.1:2333"
                      value={endpoint}
                      onChange={(event) => setEndpoint(event.target.value)}
                    />
                  </label>
                  <button
                    type="submit"
                    className="secondary"
                    aria-label={t("connectService")}
                    disabled={changing}
                  >
                    {t("connect")}
                    <ArrowRight size={14} aria-hidden />
                  </button>
                </form>
              )}
            </details>
            {needsBackendToken(error) && (
              <BackendUnlock language={state.settings.language} />
            )}
            {error != null && !needsBackendToken(error) && (
              <p role="alert" className="action-error">
                {errorText(error, state.settings.language)}
              </p>
            )}
            {setup ? (
              <>
                <BackendSettingsPanel
                  language={state.settings.language}
                  active
                  setup
                />
                <p className="backend-setup-later">{t("backendSetupLater")}</p>
              </>
            ) : (
              <div className="backend-setup-waiting">
                <p role="status">
                  {status?.error ??
                    t(
                      status
                        ? "backendStarting"
                        : error
                          ? "backendConnectHelp"
                          : "backendConnecting",
                    )}
                </p>
                {!status && !needsBackendToken(error) && (
                  <div className="backend-start-command">
                    <span>{t("backendStartCommand")}</span>
                    <code>reme start config=proactive</code>
                  </div>
                )}
              </div>
            )}
          </main>
        </div>
      </div>
    </section>
  );
}
