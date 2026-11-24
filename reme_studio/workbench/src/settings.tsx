import { useEffect, useState } from "react";
import {
  Cable,
  Palette,
  Sparkles,
  Bell,
  AudioLines,
  Info,
  Globe,
  ArrowUpRight,
  Monitor,
  Sun,
  Moon,
  Check,
} from "lucide-react";
import type { StudioState, StudioAction } from "../../client/src/protocol.ts";
import { sleepDurations } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import { Switch } from "./ui.tsx";
import VoiceSettings from "./speech/settings.tsx";
import SplitPane, { SidebarHeading } from "./split-pane.tsx";
import { githubUrl } from "./workspace-titlebar.tsx";

const sections = [
  ["general", Cable, "generalHint"],
  ["appearance", Palette, "appearanceHint"],
  ["remy", Sparkles, "remyHint"],
  ["notificationSettings", Bell, "notificationSettingsHint"],
  ["speechSettings", AudioLines, "voiceInputHint"],
  ["about", Info, "aboutHint"],
] as const;

export default function Settings({
  state,
  version,
  companionVisible,
  desktop,
  editableEndpoint,
  collapsed,
  onNavigate,
}: {
  state: StudioState;
  version: string;
  companionVisible: boolean;
  desktop: boolean;
  editableEndpoint: boolean;
  collapsed?: boolean;
  onNavigate?: () => void;
}) {
  const { action, desktop: desktopControls } = useHost();
  const t = translator(state.settings.language);
  const [section, setSection] = useState<string>("general");
  const [endpoint, setEndpoint] = useState(state.settings.endpoint);
  const [scale, setScale] = useState(state.settings.scale);
  const [error, setError] = useState<unknown>(null);
  const [saving, setSaving] = useState(false);
  useEffect(
    () => setEndpoint(state.settings.endpoint),
    [state.settings.endpoint],
  );
  useEffect(() => setScale(state.settings.scale), [state.settings.scale]);
  const run = async (value: StudioAction | (() => Promise<void>)) => {
    setError(null);
    setSaving(true);
    try {
      await (typeof value === "function" ? value() : action(value));
      return true;
    } catch (error) {
      setError(error);
      return false;
    } finally {
      setSaving(false);
    }
  };
  const items = sections.filter(([id]) => desktop || id !== "remy");
  const current = items.find(([id]) => id === section)!;
  const issue =
    error !== null
      ? errorText(error, state.settings.language)
      : section === "general" && state.connection !== "connected"
        ? state.connectionError
        : "";
  return (
    <section className="settings-layout" aria-label={t("desktopSettings")}>
      <SplitPane
        name="settings"
        path={[
          { label: t("settings"), onSelect: () => setSection("general") },
          { label: current[0] === "remy" ? "Remy" : t(current[0]) },
        ]}
        language={state.settings.language}
        collapsed={collapsed}
        sidebar={
          <>
            <SidebarHeading eyebrow="YOUR SPACE" title={t("settings")} />
            <nav className="settings-nav" aria-label={t("desktopSettings")}>
              {items.map(([id, Icon]) => (
                <button
                  key={id}
                  type="button"
                  className="settings-nav-item"
                  aria-current={section === id ? "page" : undefined}
                  onClick={() => {
                    setSection(id);
                    onNavigate?.();
                  }}
                >
                  <Icon size={16} aria-hidden />
                  {id === "remy" ? "Remy" : t(id)}
                </button>
              ))}
            </nav>
          </>
        }
      >
        <div className="settings-detail">
          <div className="settings-detail-inner">
            {section !== "about" && (
              <span className="eyebrow">PREFERENCES</span>
            )}
            <h2>{current[0] === "remy" ? "Remy" : t(current[0])}</h2>
            {section !== "about" && (
              <p className="page-description">{t(current[2])}</p>
            )}
            {issue && (
              <div className="action-error" role="alert">
                {issue}
              </div>
            )}
            {section === "general" && (
              <>
                <div className="setting-group">
                  <div className="setting-row">
                    <div>
                      <h3>{t("serviceConnection")}</h3>
                      <p>ReMe HTTP / SSE</p>
                    </div>
                    <span className="status-pill">
                      <i
                        className={`connection-dot ${state.connection === "connected" ? "online" : ""}`}
                      />
                      {t(
                        state.connection === "connected"
                          ? "connected"
                          : "reconnecting",
                      )}
                    </span>
                  </div>
                  {editableEndpoint && (
                    <form
                      className="setting-endpoint"
                      onSubmit={(event) => {
                        event.preventDefault();
                        void run({ type: "settings.update", endpoint });
                      }}
                    >
                      <label htmlFor="service-endpoint">{t("endpoint")}</label>
                      <div className="endpoint-row">
                        <input
                          id="service-endpoint"
                          aria-label={t("endpoint")}
                          value={endpoint}
                          onChange={(event) => setEndpoint(event.target.value)}
                          inputMode="url"
                          autoCapitalize="none"
                          autoCorrect="off"
                          spellCheck={false}
                        />
                        <button
                          className="primary"
                          disabled={saving || Boolean(state.pending?.streaming)}
                        >
                          {t("connect")}
                        </button>
                      </div>
                    </form>
                  )}
                </div>
                <div className="setting-group">
                  <div className="setting-row">
                    <div>
                      <h3>{t("workspaceLocation")}</h3>
                      <p className="workspace-path">
                        {state.clientInfo?.workspace_dir ?? "—"}
                      </p>
                    </div>
                    <Cable size={18} />
                  </div>
                </div>
                <p className="settings-note">{t("shortcutHint")}</p>
              </>
            )}
            {section === "appearance" && (
              <>
                <div className="setting-group appearance-group">
                  <h3>{t("theme")}</h3>
                  <div
                    className="theme-options"
                    role="group"
                    aria-label={t("theme")}
                  >
                    {(["light", "dark", "system"] as const).map((theme) => {
                      const Icon =
                        theme === "light"
                          ? Sun
                          : theme === "dark"
                            ? Moon
                            : Monitor;
                      return (
                        <button
                          key={theme}
                          aria-pressed={state.settings.theme === theme}
                          disabled={saving}
                          onClick={() =>
                            void run({ type: "settings.update", theme })
                          }
                        >
                          <div className={`theme-preview ${theme}`}>
                            <div className="mini-sidebar" />
                            <div className="mini-content">
                              <i />
                              <i />
                              <i />
                            </div>
                            {state.settings.theme === theme && (
                              <span>
                                <Check size={12} />
                              </span>
                            )}
                          </div>
                          <span>
                            <Icon size={14} />
                            {t(
                              theme === "system"
                                ? "systemTheme"
                                : theme === "light"
                                  ? "lightTheme"
                                  : "darkTheme",
                            )}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="setting-group">
                  <div className="setting-row">
                    <div>
                      <h3>{t("transparentBackground")}</h3>
                      <p>{t("transparentBackgroundHint")}</p>
                    </div>
                    <Switch
                      aria-label={t("transparentBackground")}
                      checked={state.settings.transparentBackground}
                      disabled={saving}
                      onCheckedChange={(transparentBackground) =>
                        void run({
                          type: "settings.update",
                          transparentBackground,
                        })
                      }
                    />
                  </div>
                  <div className="setting-row">
                    <div>
                      <h3>{t("interfaceLanguage")}</h3>
                      <p>
                        {t("chinese")} / {t("english")}
                      </p>
                    </div>
                    <select
                      aria-label={t("language")}
                      value={state.settings.language}
                      disabled={saving}
                      onChange={(event) =>
                        void run({
                          type: "settings.update",
                          language: event.target.value as "zh" | "en",
                        })
                      }
                    >
                      <option value="zh">{t("chinese")}</option>
                      <option value="en">{t("english")}</option>
                    </select>
                  </div>
                </div>
              </>
            )}
            {section === "remy" && desktopControls && (
              <>
                <div className="pet-preview">
                  <img src={desktopControls.companionAvatar} alt="Remy" />
                  <span>{t("companionCaption")}</span>
                </div>
                <div className="setting-group">
                  <label className="setting-range">
                    <span>
                      {t("characterSize")}
                      <strong>{Math.round(scale * 100)}%</strong>
                    </span>
                    <input
                      type="range"
                      min="0.7"
                      max="1.5"
                      step="0.05"
                      aria-label={t("characterSize")}
                      value={scale}
                      onChange={(event) => setScale(Number(event.target.value))}
                      onPointerUp={(event) =>
                        void run({
                          type: "settings.update",
                          scale: Number(event.currentTarget.value),
                        })
                      }
                      onKeyUp={(event) =>
                        void run({
                          type: "settings.update",
                          scale: Number(event.currentTarget.value),
                        })
                      }
                    />
                  </label>
                  <div className="setting-row">
                    <div>
                      <h3>{t("activityLevel")}</h3>
                      <p>{t("activityLevelHint")}</p>
                    </div>
                    <select
                      aria-label={t("activityLevel")}
                      value={state.settings.activityLevel}
                      disabled={saving}
                      onChange={(event) =>
                        void run({
                          type: "settings.update",
                          activityLevel: event.target.value as
                            "calm" | "normal" | "lively",
                        })
                      }
                    >
                      <option value="calm">{t("activityCalm")}</option>
                      <option value="normal">{t("activityNormal")}</option>
                      <option value="lively">{t("activityLively")}</option>
                    </select>
                  </div>
                  <div className="setting-row">
                    <div>
                      <h3>{t("sleepAfter")}</h3>
                      <p>{t("sleepAfterHint")}</p>
                    </div>
                    <select
                      aria-label={t("sleepAfter")}
                      value={state.settings.sleepMinutes}
                      disabled={saving}
                      onChange={(event) =>
                        void run({
                          type: "settings.update",
                          sleepMinutes: Number(event.target.value),
                        })
                      }
                    >
                      {sleepDurations.map((minutes) => (
                        <option key={minutes} value={minutes}>
                          {minutes
                            ? t("sleepAfterMinutes", { minutes })
                            : t("sleepNever")}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="setting-row">
                    <div>
                      <h3>{t("showPet")}</h3>
                    </div>
                    <Switch
                      aria-label={t("showPet")}
                      checked={companionVisible}
                      disabled={saving}
                      onCheckedChange={(visible) =>
                        void run(() =>
                          desktopControls.setCompanionVisible(visible),
                        )
                      }
                    />
                  </div>
                </div>
                <button
                  className="text-action"
                  onClick={() => void run(desktopControls.resetCompanion)}
                >
                  {t("resetPosition")}
                  <ArrowUpRight size={14} />
                </button>
              </>
            )}
            {section === "notificationSettings" && (
              <div className="setting-group">
                <div className="setting-row">
                  <div>
                    <h3>{t("notifications")}</h3>
                    <p>{t("notificationHint")}</p>
                  </div>
                  <Switch
                    aria-label={t("notifications")}
                    checked={state.settings.notifications}
                    disabled={saving}
                    onCheckedChange={(notifications) =>
                      void run({ type: "settings.update", notifications })
                    }
                  />
                </div>
                <div className="setting-row">
                  <div>
                    <h3>{t("quiet")}</h3>
                    <p>{t("quietHint")}</p>
                  </div>
                  <Switch
                    aria-label={t("quiet")}
                    checked={state.settings.quietUntil > Date.now()}
                    disabled={saving}
                    onCheckedChange={(quiet) =>
                      void run({
                        type: "settings.update",
                        quietUntil: quiet ? Date.now() + 30 * 60_000 : 0,
                      })
                    }
                  />
                </div>
              </div>
            )}
            {section === "speechSettings" && (
              <div className="setting-group voice-group">
                <VoiceSettings state={state} run={run} />
              </div>
            )}
            {section === "about" && (
              <div className="about-content">
                <div className="about-profile">
                  <span className="about-mark remy-mark" aria-hidden />
                  <div>
                    <h3>
                      Remy <span>by ReMe</span>
                    </h3>
                    <p>{t("aboutHint")}</p>
                    <span className="about-version">
                      {t("versionLabel")} {version}
                    </span>
                  </div>
                </div>
                <div className="about-links">
                  <button
                    type="button"
                    onClick={() =>
                      void run({
                        type: "link.open",
                        url: githubUrl,
                      })
                    }
                  >
                    <span className="github-mark" aria-hidden />
                    {t("openGithub")}
                    <ArrowUpRight size={12} aria-hidden />
                  </button>
                  <button
                    type="button"
                    onClick={() =>
                      void run({
                        type: "link.open",
                        url: "https://reme.agentscope.io",
                      })
                    }
                  >
                    <Globe size={18} strokeWidth={1.7} aria-hidden />
                    {t("openDocs")}
                    <ArrowUpRight size={12} aria-hidden />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </SplitPane>
    </section>
  );
}
