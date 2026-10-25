import { useState, useSyncExternalStore } from "react";
import type { ConfigForm } from "@deepseek-ai/dsh-client-ui-settings/client";

import type { ReMeSettings } from "../types.js";
import { validEndpoint } from "../endpoint.js";
import {
  ReMeStatusPage,
  statusEn,
  statusZh,
  type StatusRpc,
  type StatusTranslator,
} from "./status-page.js";
import { styles } from "./styles.js";

const NS = "reme.settings";
const STATUS_NS = "reme.status";
const SETTINGS_NS = "reme-memory";
const FIELDS = [
  "endpoint",
  "requestTimeoutMs",
  "backgroundTimeoutMs",
  "shutdownTimeoutMs",
  "autoMemoryRetryMs",
  "autoMemoryRetryMaxMs",
  "autoMemoryEnabled",
  "autoMemoryInterval",
  "autoDreamEnabled",
  "dreamCron",
  "dreamHint",
  "rootAgentsOnly",
  "language",
  "searchLimit",
  "timezone",
] as const satisfies readonly (keyof ReMeSettings)[];

type Field = (typeof FIELDS)[number];
type Translator = (key: keyof typeof en) => string;

interface ClientContext {
  get(name: string): { rpc: StatusRpc };
  effect(factory: () => (() => void) | void, label: string): void;
  locale: {
    bind(namespace: typeof NS): Translator;
    bind(namespace: typeof STATUS_NS): StatusTranslator;
    register(
      namespace: string,
      dictionaries: {
        en: Record<string, string>;
        zh: Record<string, string>;
      },
    ): () => void;
  };
  configForms: {
    get<T>(namespace: string): ConfigForm<T>;
    whileServed(namespaces: string[], register: () => () => void): () => void;
  };
  slots: {
    inject(name: string, factory: () => unknown): () => void;
    register<Props>(
      options: Record<string, unknown>,
      component: (props: Props) => JSX.Element | null,
    ): () => void;
  };
}

interface Draft {
  endpoint: string;
  requestTimeoutMs: string;
  backgroundTimeoutMs: string;
  shutdownTimeoutMs: string;
  autoMemoryRetryMs: string;
  autoMemoryRetryMaxMs: string;
  autoMemoryEnabled: boolean;
  autoMemoryInterval: string;
  autoDreamEnabled: boolean;
  dreamTime: string;
  dreamHint: string;
  rootAgentsOnly: boolean;
  language: "en" | "zh";
  searchLimit: string;
  timezone: string;
}

interface ReMeCardProps {
  scope: ConfigForm<ReMeSettings>;
  t: Translator;
  view?: "summary" | "page";
}

const en = {
  title: "ReMe Memory",
  description:
    "Turn conversations into lasting memory and grow your personal knowledge base.",
  loading: "Loading configuration…",
  unavailable: "ReMe settings are unavailable.",
  connected: "Connected",
  disconnected: "Unavailable",
  unchecked: "Not checked",
  refresh: "Refresh status",
  website: "ReMe website",
  open: "Open ReMe",
  runDream: "Consolidate Memory Now",
  connection: "Connection and search",
  endpoint: "Service URL",
  endpointHint: "The ReMe HTTP service used by DSH.",
  language: "Guidance language",
  searchLimit: "Default search results",
  requestTimeout: "Search timeout (ms)",
  autoMemory: "Automatic memory",
  autoMemoryEnabled: "Capture completed conversations automatically",
  autoMemoryInterval: "Submit every N completed turns",
  rootOnly: "Exclude subagents",
  timezone: "Workspace timezone",
  autoDream: "Memory consolidation",
  autoDreamEnabled: "Consolidate long-term memory every day",
  dreamTime: "Daily consolidation time",
  dreamHint: "Consolidation guidance",
  advanced: "Advanced",
  backgroundTimeout: "Background request timeout (ms)",
  shutdownTimeout: "Shutdown flush timeout (ms)",
  retryDelay: "Retry delay (ms)",
  retryMaxDelay: "Maximum retry delay (ms)",
  diagnostics: "Service diagnostics",
  processMemory: "Process memory",
  componentMemory: "Component memory",
  components: "Health components",
  componentInstances: "Component details",
  noInstances: "No configured instance",
  started: "Started",
  notStarted: "Not started",
  instance: "Instance",
  fileGraph: "File graph",
  fileStore: "File store",
  keywordIndex: "Keyword index",
  embeddingStore: "Embedding store",
  modelName: "Model",
  dimensions: "Dimensions",
  cacheSize: "Cache entries",
  nodes: "Nodes",
  edges: "Edges",
  virtualNodes: "Virtual nodes",
  pendingNodes: "Pending nodes",
  chunks: "Chunks",
  embeddedChunks: "Embedded chunks",
  documents: "Documents",
  vocabulary: "Vocabulary",
  memoryUsage: "Memory",
  serverConfig: "Server configuration (redacted)",
  loadConfig: "Load server configuration",
  save: "Save",
  saving: "Saving…",
  discard: "Discard",
  reset: "Reset to deployment defaults",
  unsaved: "Unsaved",
  invalid: "Fix the invalid fields before saving.",
  saveFailed: "The server did not accept all settings.",
  saved: "Settings saved.",
  dreamComplete: "Memory consolidation completed.",
  dreamFailed: "Memory consolidation failed",
  readOnly: "The settings document is read-only.",
  version: "Version",
  healthy: "Healthy",
  unhealthy: "Unhealthy",
};

const zh: typeof en = {
  title: "ReMe Memory",
  description: "让对话沉淀为长期记忆，持续构建你的个人知识库。",
  loading: "正在加载配置…",
  unavailable: "ReMe 设置不可用。",
  connected: "已连接",
  disconnected: "连接失败",
  unchecked: "尚未检查",
  refresh: "刷新状态",
  website: "ReMe 官网",
  open: "打开 ReMe",
  runDream: "立即整理",
  connection: "连接与搜索",
  endpoint: "服务地址",
  endpointHint: "DSH 访问的 ReMe HTTP 服务。",
  language: "记忆指引语言",
  searchLimit: "默认搜索数量",
  requestTimeout: "搜索超时（毫秒）",
  autoMemory: "自动记忆",
  autoMemoryEnabled: "自动记录已完成的对话",
  autoMemoryInterval: "每 N 个完成回合提交一次",
  rootOnly: "排除子 Agent",
  timezone: "Workspace 时区",
  autoDream: "记忆整理",
  autoDreamEnabled: "每天自动整理长期记忆",
  dreamTime: "每日整理时间",
  dreamHint: "记忆整理指引",
  advanced: "高级设置",
  backgroundTimeout: "后台请求超时（毫秒）",
  shutdownTimeout: "关闭时写入等待（毫秒）",
  retryDelay: "失败重试间隔（毫秒）",
  retryMaxDelay: "最大重试间隔（毫秒）",
  diagnostics: "服务诊断",
  processMemory: "进程内存",
  componentMemory: "组件内存",
  components: "健康组件",
  componentInstances: "组件详情",
  noInstances: "未配置实例",
  started: "已启动",
  notStarted: "未启动",
  instance: "实例",
  fileGraph: "文件图谱",
  fileStore: "文件存储",
  keywordIndex: "关键词索引",
  embeddingStore: "向量存储",
  modelName: "模型",
  dimensions: "向量维度",
  cacheSize: "缓存条目",
  nodes: "节点",
  edges: "边",
  virtualNodes: "虚拟节点",
  pendingNodes: "待处理节点",
  chunks: "内容块",
  embeddedChunks: "已生成向量",
  documents: "文档",
  vocabulary: "词汇量",
  memoryUsage: "内存占用",
  serverConfig: "服务端配置（已脱敏）",
  loadConfig: "读取服务端配置",
  save: "保存",
  saving: "正在保存…",
  discard: "放弃修改",
  reset: "恢复部署默认值",
  unsaved: "未保存",
  invalid: "请先修正无效字段。",
  saveFailed: "服务端未接受全部设置。",
  saved: "设置已保存。",
  dreamComplete: "记忆整理已完成。",
  dreamFailed: "记忆整理失败",
  readOnly: "设置文件当前只读。",
  version: "版本",
  healthy: "健康",
  unhealthy: "异常",
};

export const inject = ["slots", "locale", "configForms", "connection"];

export function apply(ctx: ClientContext): void {
  const t = ctx.locale.bind(NS);
  ctx.effect(
    () => ctx.locale.register(NS, { en, zh }),
    "remeMemory.settingsLocale()",
  );
  ctx.effect(
    () => ctx.locale.register(STATUS_NS, { en: statusEn, zh: statusZh }),
    "remeMemory.statusLocale()",
  );
  ctx.effect(() => installStyles(), "remeMemory.settingsStyles()");
  const scope = ctx.configForms.get<ReMeSettings>(SETTINGS_NS);
  ctx.effect(
    () =>
      ctx.configForms.whileServed([SETTINGS_NS], () =>
        ctx.slots.inject("plugins.row.config", () =>
          ctx.slots.register(
            {
              name: "plugins.row.config",
              key: "@agentscope-ai/reme-dsh-plugin#reme-memory",
              label: () => t("title"),
              locale: NS,
              inject: () => ({ scope, t }),
            },
            ReMeSettingsCard,
          ),
        ),
      ),
    "remeMemory.settingsCard()",
  );
  const statusT = ctx.locale.bind(STATUS_NS);
  const { rpc } = ctx.get("connection");
  ctx.slots.inject("settings.section", () =>
    ctx.slots.register(
      {
        name: "settings.section",
        id: "reme-status",
        order: 30,
        label: () => statusT("nav"),
        meta: { icon: "memory" },
        locale: STATUS_NS,
        inject: () => ({ scope, rpc, t: statusT }),
      },
      ReMeStatusPage,
    ),
  );
}

function ReMeSettingsCard({
  scope,
  t,
  view,
}: ReMeCardProps): JSX.Element | null {
  const snapshot = useSyncExternalStore(
    (listener) => scope.subscribe(listener),
    () => scope.getSnapshot(),
  );
  const [draftOverride, setDraft] = useState<{
    value: Draft;
    revision: number | undefined;
  }>();
  const [resetAll, setResetAll] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveState, setSaveState] = useState<"" | "saved" | "failed">("");
  const value = snapshot.value;
  const draft =
    draftOverride?.value ??
    (value === undefined ? undefined : draftFrom(value));

  if (view === "summary") return <>{t("description")}</>;
  if (snapshot.status === "unavailable") return null;
  const dirty =
    resetAll ||
    (value !== undefined &&
      draft !== undefined &&
      !sameDraft(draft, draftFrom(value)));
  const validation = draft === undefined ? undefined : parseDraft(draft);

  const update = <K extends keyof Draft>(field: K, next: Draft[K]) => {
    if (value === undefined) return;
    setDraft((current) => ({
      value: { ...(current?.value ?? draftFrom(value)), [field]: next },
      revision: current ? current.revision : snapshot.revision,
    }));
    setResetAll(false);
    setSaveState("");
  };

  const save = async () => {
    if (
      value === undefined ||
      draft === undefined ||
      validation === undefined ||
      !snapshot.writable ||
      !dirty ||
      saving
    )
      return;
    setSaving(true);
    setSaveState("");
    try {
      const ops = resetAll
        ? FIELDS.map((field) => ({ op: "unset" as const, path: [field] }))
        : FIELDS.flatMap((field) =>
            Object.is(value[field], validation[field])
              ? []
              : [
                  {
                    op: "set" as const,
                    path: [field],
                    value: validation[field],
                  },
                ],
          );
      const success = await scope.mutate(
        ops,
        draftOverride?.revision ?? snapshot.revision,
      );
      setSaveState(success ? "saved" : "failed");
      if (success) {
        setDraft(undefined);
        setResetAll(false);
      }
    } catch {
      setSaveState("failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="reme-settings-card">
      <div className="reme-settings-body">
        {dirty ? <p className="reme-settings-pending">{t("unsaved")}</p> : null}
        {snapshot.status === "loading" ||
        draft === undefined ||
        value === undefined ? (
          <p className="reme-settings-muted">{t("loading")}</p>
        ) : (
          <>
            <fieldset
              className="reme-settings-fields"
              disabled={saving || !snapshot.writable}
            >
              <SettingsForm draft={draft} update={update} t={t} />
            </fieldset>
            {!snapshot.writable ? (
              <p className="reme-settings-error" role="status">
                {t("readOnly")}
              </p>
            ) : null}
            {validation === undefined ? (
              <p className="reme-settings-error" role="status">
                {t("invalid")}
              </p>
            ) : null}
            <div className="reme-settings-actions">
              <button
                type="button"
                className="reme-settings-reset"
                disabled={!snapshot.writable || saving}
                onClick={() => {
                  setDraft({
                    value: draftFrom(baseSettings(snapshot.base, value)),
                    revision: snapshot.revision,
                  });
                  setResetAll(true);
                  setSaveState("");
                }}
              >
                {t("reset")}
              </button>
              <span className="reme-settings-result" role="status">
                {saveState === "failed" ? (
                  <span className="reme-settings-error">{t("saveFailed")}</span>
                ) : null}
                {saveState === "saved" ? (
                  <span className="reme-settings-success">{t("saved")}</span>
                ) : null}
              </span>
              <button
                type="button"
                className="reme-settings-discard"
                disabled={!dirty || saving}
                onClick={() => {
                  setDraft(undefined);
                  setResetAll(false);
                  setSaveState("");
                }}
              >
                {t("discard")}
              </button>
              <button
                type="button"
                className="reme-settings-save"
                disabled={
                  !snapshot.writable ||
                  !dirty ||
                  validation === undefined ||
                  saving
                }
                onClick={() => void save()}
              >
                {t(saving ? "saving" : "save")}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function SettingsForm({
  draft,
  update,
  t,
}: {
  draft: Draft;
  update: <K extends keyof Draft>(field: K, value: Draft[K]) => void;
  t: Translator;
}): JSX.Element {
  return (
    <>
      <section className="reme-settings-section">
        <h4>{t("connection")}</h4>
        <div className="reme-settings-grid">
          <Field wide label={t("endpoint")} hint={t("endpointHint")}>
            <input
              value={draft.endpoint}
              aria-invalid={!validEndpoint(draft.endpoint)}
              onChange={(event) => update("endpoint", event.target.value)}
            />
          </Field>
          <Field label={t("language")}>
            <select
              value={draft.language}
              onChange={(event) =>
                update("language", event.target.value as "en" | "zh")
              }
            >
              <option value="zh">中文</option>
              <option value="en">English</option>
            </select>
          </Field>
          <Field label={t("searchLimit")}>
            <input
              type="number"
              min="1"
              max="50"
              value={draft.searchLimit}
              onChange={(event) => update("searchLimit", event.target.value)}
            />
          </Field>
          <Field label={t("requestTimeout")}>
            <input
              type="number"
              min="1000"
              max="120000"
              value={draft.requestTimeoutMs}
              onChange={(event) =>
                update("requestTimeoutMs", event.target.value)
              }
            />
          </Field>
        </div>
      </section>
      <section className="reme-settings-section">
        <h4>{t("autoMemory")}</h4>
        <div className="reme-settings-grid">
          <Check
            label={t("autoMemoryEnabled")}
            checked={draft.autoMemoryEnabled}
            onChange={(value) => update("autoMemoryEnabled", value)}
          />
          <Check
            label={t("rootOnly")}
            checked={draft.rootAgentsOnly}
            onChange={(value) => update("rootAgentsOnly", value)}
          />
          <Field label={t("autoMemoryInterval")}>
            <input
              type="number"
              min="1"
              max="1000"
              value={draft.autoMemoryInterval}
              onChange={(event) =>
                update("autoMemoryInterval", event.target.value)
              }
            />
          </Field>
          <Field label={t("timezone")}>
            <input
              value={draft.timezone}
              aria-invalid={!validTimezone(draft.timezone)}
              onChange={(event) => update("timezone", event.target.value)}
            />
          </Field>
        </div>
      </section>
      <section className="reme-settings-section">
        <h4>{t("autoDream")}</h4>
        <div className="reme-settings-grid">
          <Check
            label={t("autoDreamEnabled")}
            checked={draft.autoDreamEnabled}
            onChange={(value) => update("autoDreamEnabled", value)}
          />
          <Field label={t("dreamTime")}>
            <input
              type="time"
              value={draft.dreamTime}
              onChange={(event) => update("dreamTime", event.target.value)}
            />
          </Field>
          <Field wide label={t("dreamHint")}>
            <textarea
              value={draft.dreamHint}
              onChange={(event) => update("dreamHint", event.target.value)}
            />
          </Field>
        </div>
      </section>
      <section className="reme-settings-section">
        <h4>{t("advanced")}</h4>
        <div className="reme-settings-grid">
          <Field label={t("backgroundTimeout")}>
            <input
              type="number"
              min="1000"
              max="3600000"
              value={draft.backgroundTimeoutMs}
              onChange={(event) =>
                update("backgroundTimeoutMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("retryDelay")}>
            <input
              type="number"
              min={1000}
              max={60000}
              value={draft.autoMemoryRetryMs}
              onChange={(event) =>
                update("autoMemoryRetryMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("retryMaxDelay")}>
            <input
              type="number"
              min={1000}
              max={3600000}
              value={draft.autoMemoryRetryMaxMs}
              onChange={(event) =>
                update("autoMemoryRetryMaxMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("shutdownTimeout")}>
            <input
              type="number"
              min="100"
              max="60000"
              value={draft.shutdownTimeoutMs}
              onChange={(event) =>
                update("shutdownTimeoutMs", event.target.value)
              }
            />
          </Field>
        </div>
      </section>
    </>
  );
}

function Field({
  label,
  hint,
  wide = false,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: JSX.Element;
}): JSX.Element {
  return (
    <div className={`reme-settings-field${wide ? " wide" : ""}`}>
      <label>{label}</label>
      {children}
      {hint ? <small>{hint}</small> : null}
    </div>
  );
}

function Check({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange(value: boolean): void;
}): JSX.Element {
  return (
    <label className="reme-settings-check">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

function draftFrom(value: ReMeSettings): Draft {
  return {
    endpoint: value.endpoint,
    requestTimeoutMs: String(value.requestTimeoutMs),
    backgroundTimeoutMs: String(value.backgroundTimeoutMs),
    shutdownTimeoutMs: String(value.shutdownTimeoutMs),
    autoMemoryRetryMs: String(value.autoMemoryRetryMs),
    autoMemoryRetryMaxMs: String(value.autoMemoryRetryMaxMs),
    autoMemoryEnabled: value.autoMemoryEnabled,
    autoMemoryInterval: String(value.autoMemoryInterval),
    autoDreamEnabled: value.autoDreamEnabled,
    dreamTime: timeFromCron(value.dreamCron),
    dreamHint: value.dreamHint,
    rootAgentsOnly: value.rootAgentsOnly,
    language: value.language,
    searchLimit: String(value.searchLimit),
    timezone: value.timezone,
  };
}

function parseDraft(draft: Draft): ReMeSettings | undefined {
  const requestTimeoutMs = boundedInteger(draft.requestTimeoutMs, 1000, 120000);
  const backgroundTimeoutMs = boundedInteger(
    draft.backgroundTimeoutMs,
    1000,
    3600000,
  );
  const shutdownTimeoutMs = boundedInteger(draft.shutdownTimeoutMs, 100, 60000);
  const autoMemoryRetryMs = boundedInteger(
    draft.autoMemoryRetryMs,
    1000,
    60000,
  );
  const autoMemoryRetryMaxMs = boundedInteger(
    draft.autoMemoryRetryMaxMs,
    autoMemoryRetryMs ?? 1000,
    3600000,
  );
  const autoMemoryInterval = boundedInteger(draft.autoMemoryInterval, 1, 1000);
  const searchLimit = boundedInteger(draft.searchLimit, 1, 50);
  if (
    !validEndpoint(draft.endpoint) ||
    !validTimezone(draft.timezone) ||
    !/^\d{2}:\d{2}$/.test(draft.dreamTime)
  )
    return undefined;
  if (
    [
      requestTimeoutMs,
      backgroundTimeoutMs,
      shutdownTimeoutMs,
      autoMemoryRetryMs,
      autoMemoryRetryMaxMs,
      autoMemoryInterval,
      searchLimit,
    ].some((value) => value === undefined)
  )
    return undefined;
  const [hour, minute] = draft.dreamTime.split(":");
  return {
    endpoint: draft.endpoint.trim().replace(/\/+$/, ""),
    requestTimeoutMs: requestTimeoutMs!,
    backgroundTimeoutMs: backgroundTimeoutMs!,
    shutdownTimeoutMs: shutdownTimeoutMs!,
    autoMemoryRetryMs: autoMemoryRetryMs!,
    autoMemoryRetryMaxMs: autoMemoryRetryMaxMs!,
    autoMemoryEnabled: draft.autoMemoryEnabled,
    autoMemoryInterval: autoMemoryInterval!,
    autoDreamEnabled: draft.autoDreamEnabled,
    dreamCron: `${Number(minute)} ${Number(hour)} * * *`,
    dreamHint: draft.dreamHint,
    rootAgentsOnly: draft.rootAgentsOnly,
    language: draft.language,
    searchLimit: searchLimit!,
    timezone: draft.timezone.trim(),
  };
}

function boundedInteger(
  text: string,
  minimum: number,
  maximum: number,
): number | undefined {
  const value = Number(text);
  return Number.isInteger(value) && value >= minimum && value <= maximum
    ? value
    : undefined;
}

function validTimezone(value: string): boolean {
  try {
    new Intl.DateTimeFormat("en", { timeZone: value.trim() }).format(0);
    return value.trim().length > 0;
  } catch {
    return false;
  }
}

function timeFromCron(cron: string): string {
  const match = /^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*$/.exec(cron.trim());
  if (match === null) return "";
  return `${String(Number(match[2])).padStart(2, "0")}:${String(
    Number(match[1]),
  ).padStart(2, "0")}`;
}

function baseSettings(base: unknown, fallback: ReMeSettings): ReMeSettings {
  return isRecord(base) ? (base as unknown as ReMeSettings) : fallback;
}

function sameDraft(left: Draft, right: Draft): boolean {
  return (Object.keys(left) as (keyof Draft)[]).every((key) =>
    Object.is(left[key], right[key]),
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function installStyles(): () => void {
  const tag = document.createElement("style");
  tag.dataset.pluginCss = "@agentscope-ai/reme-dsh-plugin/settings";
  tag.textContent = styles;
  document.head.appendChild(tag);
  return () => tag.remove();
}
