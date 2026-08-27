"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  BadgeInfo,
  Braces,
  Check,
  CircleAlert,
  Cpu,
  DatabaseZap,
  FileArchive,
  GitFork,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  X,
} from "lucide-react";
import {
  getAppConfig,
  getReMeHealth,
  getReMeStatus,
  getReMeVersion,
  rebuildReMeIndex,
  REME_API_ENDPOINT,
} from "./api";
import { healthComponentEntries, isComponentHealthy } from "./health-status";
import { useI18n, type TranslationKey } from "./i18n";
import type {
  AppConfig,
  ReMeComponentHealth,
  ReMeHealth,
  ReMeResponse,
} from "./types";

type SettingsSection = "status" | "index" | "config" | "version";

interface MemoryUsage {
  bytes?: number;
  human?: string;
}

interface MemoryStatus {
  components?: Record<string, Record<string, MemoryUsage>>;
  components_total?: string;
  process_rss?: string;
}

function memoryFrom(response?: ReMeResponse<string>): MemoryStatus | undefined {
  const status = response?.metadata.status;
  if (!status || typeof status !== "object") return undefined;
  const memory = (status as Record<string, unknown>).memory;
  return memory && typeof memory === "object"
    ? (memory as MemoryStatus)
    : undefined;
}

const COMPONENT_ICONS: Record<string, React.ReactNode> = {
  embedding_store: <Cpu size={18} />,
  file_graph: <GitFork size={18} />,
  file_store: <HardDrive size={18} />,
  keyword_index: <FileArchive size={18} />,
};

const COMPONENT_LABELS = {
  embedding_store: "embeddingStore",
  file_graph: "fileGraph",
  file_store: "fileStore",
  keyword_index: "keywordIndex",
} as const;

const COMPONENT_FIELDS = {
  model_name: "modelName",
  dimensions: "dimensions",
  cache_size: "cacheSize",
  n_nodes: "nodes",
  n_edges: "edges",
  n_virtual: "virtualNodes",
  n_pending: "pendingNodes",
  n_chunks: "chunks",
  n_chunks_with_embedding: "embeddedChunks",
  n_docs: "documents",
  vocab_size: "vocabulary",
} as const;

function ComponentCard({
  type,
  name,
  component,
  memory,
}: {
  type: string;
  name: string;
  component: ReMeComponentHealth;
  memory?: string;
}) {
  const { t } = useI18n();
  const healthy = isComponentHealthy(component);
  const facts: Array<{ label: TranslationKey; value: unknown }> =
    Object.entries(COMPONENT_FIELDS).flatMap(([field, label]) => {
      const value = component[field as keyof ReMeComponentHealth];
      return value === undefined || value === null
        ? []
        : [{ label: label as TranslationKey, value }];
    });
  if (memory || component.memory) {
    facts.push({
      label: "memoryUsage",
      value: memory || component.memory || "—",
    });
  }
  const label = COMPONENT_LABELS[type as keyof typeof COMPONENT_LABELS];

  return (
    <article className={`component-card ${healthy ? "healthy" : "unhealthy"}`}>
      <header>
        <span>{COMPONENT_ICONS[type] || <Cpu size={18} />}</span>
        <div>
          <strong>{label ? t(label) : type}</strong>
          <small>{name === "default" ? t("defaultInstance") : name}</small>
        </div>
        <span className={`component-health ${healthy ? "" : "unhealthy"}`}>
          <i />
          {t(healthy ? "healthy" : "unhealthy")}
        </span>
      </header>
      {facts.length > 0 && (
        <dl>
          {facts.map(({ label: factLabel, value }) => (
            <div key={factLabel}>
              <dt>{t(factLabel)}</dt>
              <dd>{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </article>
  );
}

function SettingsState({
  loading,
  error,
}: {
  loading: boolean;
  error?: string;
}) {
  const { t } = useI18n();
  if (loading)
    return (
      <div className="settings-state">
        <LoaderCircle className="spin" size={18} />
        {t("loadingSettings")}
      </div>
    );
  if (error)
    return (
      <div className="settings-state error">
        <CircleAlert size={18} />
        {error}
      </div>
    );
  return null;
}

export default function SettingsCenter({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const [section, setSection] = useState<SettingsSection>("status");
  const [status, setStatus] = useState<ReMeResponse<string>>();
  const [health, setHealth] = useState<ReMeHealth>();
  const [config, setConfig] = useState<AppConfig>();
  const [version, setVersion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [confirmingIndex, setConfirmingIndex] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [indexResult, setIndexResult] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [nextStatus, nextHealth, nextConfig, nextVersion] =
        await Promise.all([
          getReMeStatus(),
          getReMeHealth().catch(() => undefined),
          getAppConfig(),
          getReMeVersion(),
        ]);
      setStatus(nextStatus);
      setHealth(nextHealth);
      setConfig(nextConfig);
      setVersion(nextVersion);
    } catch (nextError) {
      setError(
        nextError instanceof Error ? nextError.message : "ReMe unavailable",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const frame = requestAnimationFrame(() => void load());
    return () => cancelAnimationFrame(frame);
  }, [load, open]);
  useEffect(() => {
    if (!open) return;
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose, open]);

  const rebuild = async () => {
    setConfirmingIndex(false);
    setReindexing(true);
    setIndexResult("");
    setError("");
    try {
      const result = await rebuildReMeIndex();
      setIndexResult(
        typeof result.answer === "string" && result.answer
          ? result.answer
          : t("indexRebuilt"),
      );
      const nextStatus = await getReMeStatus();
      setStatus(nextStatus);
    } catch (nextError) {
      setError(
        nextError instanceof Error
          ? nextError.message
          : t("requestFailed", { status: "" }),
      );
    } finally {
      setReindexing(false);
    }
  };

  const memory = memoryFrom(status);
  const components = Object.entries(memory?.components || {}).flatMap(
    ([type, entries]) =>
      Object.entries(entries).map(([name, usage]) => ({ type, name, usage })),
  );
  const healthComponents = healthComponentEntries(health, memory?.components);
  const healthyComponents = healthComponents.filter(({ component }) =>
    isComponentHealthy(component),
  ).length;
  const sections: Array<{
    id: SettingsSection;
    label: string;
    icon: React.ReactNode;
  }> = [
    { id: "status", label: t("settingsStatus"), icon: <Activity size={18} /> },
    { id: "index", label: t("settingsIndex"), icon: <DatabaseZap size={18} /> },
    { id: "config", label: t("settingsConfig"), icon: <Braces size={18} /> },
    {
      id: "version",
      label: t("settingsVersion"),
      icon: <BadgeInfo size={18} />,
    },
  ];

  return (
    <div
      className={`settings-overlay ${open ? "open" : ""}`}
      aria-hidden={!open}
      inert={!open}
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="settings-window"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
      >
        <header className="settings-header">
          <div>
            <h2 id="settings-title">{t("settingsTitle")}</h2>
            <p>{t("settingsDescription")}</p>
          </div>
          <button
            onClick={onClose}
            aria-label={t("closeSettings")}
            title={t("closeSettings")}
          >
            <X size={18} />
          </button>
        </header>
        <div className="settings-layout">
          <nav className="settings-sidebar" aria-label={t("settingsTitle")}>
            {sections.map((item) => (
              <button
                key={item.id}
                className={section === item.id ? "active" : ""}
                onClick={() => {
                  setSection(item.id);
                  setError("");
                }}
              >
                <span>{item.icon}</span>
                {item.label}
              </button>
            ))}
          </nav>
          <main className="settings-content">
            {section !== "index" && (
              <button
                className="settings-refresh"
                onClick={() => void load()}
                disabled={loading}
              >
                <RefreshCw className={loading ? "spin" : ""} size={15} />
                {t("refresh")}
              </button>
            )}
            <SettingsState
              loading={loading && !status && !health && !config && !version}
              error={section === "index" ? undefined : error}
            />
            {section === "status" && status && (
              <div className="settings-page">
                <section className="status-hero">
                  <div className="status-hero-brand">
                    <span className="settings-page-icon status">
                      <Activity size={25} />
                    </span>
                    <div>
                      <small>ReMe Memory</small>
                      <h3>{t("serviceOverview")}</h3>
                      <code>{REME_API_ENDPOINT}</code>
                    </div>
                  </div>
                  <div className="status-hero-health">
                    <span
                      className={`health-badge ${
                        health?.healthy === false ? "unhealthy" : ""
                      }`}
                    >
                      <i />
                      {health?.healthy === false
                        ? t("serviceAttention")
                        : t("serviceOnline")}
                    </span>
                    <strong>v{health?.version || version || "—"}</strong>
                    <small>{t("currentVersion")}</small>
                  </div>
                </section>
                <div className="status-metrics">
                  <article>
                    <small>{t("processMemory")}</small>
                    <strong>{memory?.process_rss || "—"}</strong>
                  </article>
                  <article>
                    <small>{t("componentMemory")}</small>
                    <strong>{memory?.components_total || "—"}</strong>
                  </article>
                  <article>
                    <small>{t("healthyComponents")}</small>
                    <strong>
                      {healthyComponents}
                      <span> / {healthComponents.length}</span>
                    </strong>
                  </article>
                </div>
                <section className="component-section">
                  <div className="component-section-title">
                    <div>
                      <h4>{t("componentDetails")}</h4>
                      <p>{t("componentDetailsDescription")}</p>
                    </div>
                  </div>
                  <div className="component-grid">
                    {healthComponents.map(
                      ({ type, name, component, memory: componentMemory }) => (
                        <ComponentCard
                          key={`${type}:${name}`}
                          type={type}
                          name={name}
                          component={component}
                          memory={componentMemory}
                        />
                      ),
                    )}
                    {!healthComponents.length &&
                      components.map(({ type, name, usage }) => (
                        <article
                          className="component-card"
                          key={`${type}:${name}`}
                        >
                          <header>
                            <span>
                              {COMPONENT_ICONS[type] || <Cpu size={18} />}
                            </span>
                            <div>
                              <strong>{type}</strong>
                              <small>{name}</small>
                            </div>
                          </header>
                          <dl>
                            <div>
                              <dt>{t("memoryUsage")}</dt>
                              <dd>{usage.human || "—"}</dd>
                            </div>
                          </dl>
                        </article>
                      ))}
                  </div>
                </section>
              </div>
            )}
            {section === "index" && (
              <div className="settings-page">
                <div className="settings-page-title">
                  <span className="settings-page-icon index">
                    <DatabaseZap size={25} />
                  </span>
                  <div>
                    <h3>{t("indexTitle")}</h3>
                    <p>{t("indexDescription")}</p>
                  </div>
                </div>
                <section className="settings-card index-card">
                  {!confirmingIndex ? (
                    <button
                      className="primary-action"
                      onClick={() => setConfirmingIndex(true)}
                      disabled={reindexing}
                    >
                      {reindexing ? (
                        <LoaderCircle className="spin" size={16} />
                      ) : (
                        <DatabaseZap size={16} />
                      )}
                      {reindexing ? t("rebuildingIndex") : t("rebuildIndex")}
                    </button>
                  ) : (
                    <div className="index-confirm">
                      <div>
                        <strong>{t("confirmReindexTitle")}</strong>
                        <p>{t("confirmReindexDescription")}</p>
                      </div>
                      <div>
                        <button onClick={() => setConfirmingIndex(false)}>
                          {t("cancel")}
                        </button>
                        <button
                          className="danger-action"
                          onClick={() => void rebuild()}
                        >
                          {t("confirmReindex")}
                        </button>
                      </div>
                    </div>
                  )}
                  {indexResult && (
                    <div className="settings-success">
                      <Check size={16} />
                      {indexResult}
                    </div>
                  )}
                  {error && (
                    <div className="settings-state error">
                      <CircleAlert size={18} />
                      {error}
                    </div>
                  )}
                </section>
              </div>
            )}
            {section === "config" && config && (
              <div className="settings-page">
                <div className="settings-page-title">
                  <span className="settings-page-icon config">
                    <Braces size={25} />
                  </span>
                  <div>
                    <h3>{t("effectiveConfig")}</h3>
                    <p>{t("redactedConfig")}</p>
                  </div>
                </div>
                <pre className="config-json">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </div>
            )}
            {section === "version" && version && (
              <div className="settings-page">
                <div className="settings-page-title">
                  <span className="settings-page-icon version">
                    <BadgeInfo size={25} />
                  </span>
                  <div>
                    <h3>{t("settingsVersion")}</h3>
                    <p>{t("currentVersion")}</p>
                  </div>
                </div>
                <section className="settings-card version-card">
                  <div>
                    <small>ReMe</small>
                    <strong>v{version}</strong>
                  </div>
                  <div>
                    <small>{t("apiEndpoint")}</small>
                    <code>{REME_API_ENDPOINT}</code>
                  </div>
                </section>
              </div>
            )}
          </main>
        </div>
      </section>
    </div>
  );
}
