import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { SettingsFormModel } from "@deepseek-ai/dsh-client-ui-primitives";
import type {
  SettingsFieldState,
  SettingsFormShell,
} from "@deepseek-ai/dsh-client-ui-primitives";
import type { ConfigForm } from "@deepseek-ai/dsh-client-ui-settings/client";

import type { ReMeSettings } from "../types.js";
import {
  ReMeEnvironmentTab,
  envEn,
  envZh,
  type EnvTranslator,
} from "../environment/client/tab.js";
import { environmentStyles } from "../environment/client/styles.js";
import {
  ReMeStatusPage,
  statusEn,
  statusZh,
  type StatusRpc,
  type StatusTranslator,
} from "./status-page.js";
import { SETTINGS_SPECS } from "./settings-specs.js";
import { styles } from "./styles.js";

const NS = "reme.settings";
const STATUS_NS = "reme.status";
const ENV_NS = "reme.environment";
const SETTINGS_NS = "reme-memory";

type Translator = (key: keyof typeof en) => string;

interface ClientContext {
  get(name: string): { rpc: StatusRpc };
  effect(factory: () => void | (() => void), label: string): void;
  locale: {
    bind(namespace: typeof NS): Translator;
    bind(namespace: typeof STATUS_NS): StatusTranslator;
    bind(namespace: typeof ENV_NS): EnvTranslator;
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

/** One control as the model currently stages it. */
type FieldState = SettingsFieldState;

/**
 * The slice of DSH's snapshot store this card reads.
 *
 * `@deepseek-ai/dsh-client-store` is not a dependency of this package, so the
 * store's own type is not available here; the card declares the two members it
 * uses instead of depending on the package.
 */
interface BoundStore<T> {
  getSnapshot(): T;
  subscribe(listener: () => void): () => void;
}

/** Everything the card renders, rebuilt from the model on every publish. */
interface CardState {
  /** Whether the Host still serves this namespace to this client. */
  status: "loading" | "ready" | "unavailable";
  /** What a save would do, and whether one is crossing the wire. */
  shell: SettingsFormShell;
  /** Every control's staged state, keyed by field name. */
  fields: Record<string, FieldState>;
}

/**
 * Rebuild the card's view state from the model.
 *
 * The model publishes when the scope changes, when a draft is staged, when a
 * save starts and when it settles, so this projection is the card's only
 * source of truth: what it renders is what the model holds.
 *
 * The projection carries the model's own state and nothing the settings
 * service owns: the store freezes what it is given, and the resolved section
 * belongs to the service, which replaces it rather than editing it.
 */
function projectCard(
  model: SettingsFormModel<ReMeSettings>,
  scope: ConfigForm<ReMeSettings>,
): CardState {
  const fields: Record<string, FieldState> = {};
  for (const spec of SETTINGS_SPECS)
    fields[spec.field] = model.field(spec.field);
  return {
    status: scope.getSnapshot().status,
    shell: model.shell(),
    fields,
  };
}

interface ReMeCardProps {
  scope: ConfigForm<ReMeSettings>;
  t: Translator;
  rpc?: StatusRpc;
  envT?: EnvTranslator;
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
  tabMemory: "Memory settings",
  tabEnvironment: "Environment",
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
  tabMemory: "记忆设置",
  tabEnvironment: "环境管理",
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
  ctx.effect(
    () => ctx.locale.register(ENV_NS, { en: envEn, zh: envZh }),
    "remeMemory.envLocale()",
  );
  ctx.effect(() => installStyles(), "remeMemory.settingsStyles()");
  const envT = ctx.locale.bind(ENV_NS);
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
              inject: () => ({
                scope,
                t,
                rpc: ctx.get("connection").rpc,
                envT,
              }),
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
  rpc,
  envT,
  view,
}: ReMeCardProps): JSX.Element | null {
  // DSH's form model stages the drafts, marks the ones that are invalid and
  // writes every staged edit as one revision-fenced mutation, so this page owns
  // only its controls and the field specs behind them.
  //
  // The model, its actions and the store it publishes to are built once: the
  // model owns one subscription to the scope, and `bind` registers one
  // listener on the model, so a second build would publish every change twice.
  const [form] = useState(() => {
    const model = new SettingsFormModel<ReMeSettings>(scope, SETTINGS_SPECS);
    return {
      model,
      actions: model.actions(),
      store: model.bind(() =>
        projectCard(model, scope),
      ) as BoundStore<CardState>,
    };
  });
  // The model publishes on a scope change, on every staged edit, when a save
  // starts and when it settles. The card reads those publications and nothing
  // else, so a staged edit repaints the controls on its own.
  const state = useSyncExternalStore(
    form.store.subscribe,
    form.store.getSnapshot,
  );
  const disposeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  );
  const [saved, setSaved] = useState(false);
  const [tab, setTab] = useState<"memory" | "environment">("memory");
  useEffect(() => {
    // React StrictMode mounts, cleans up and mounts again in development.
    // Disposing the model immediately would hand the remounted card a dead
    // one, so the cleanup is deferred and cancelled by the next mount.
    if (disposeTimer.current !== undefined) {
      clearTimeout(disposeTimer.current);
      disposeTimer.current = undefined;
    }
    return () => {
      disposeTimer.current = setTimeout(() => {
        disposeTimer.current = undefined;
        form.model.dispose();
      }, 0);
    };
  }, [form.model]);

  if (view === "summary") return <>{t("description")}</>;
  if (state.status === "unavailable") return null;

  const { shell, fields } = state;
  // The model's own saving state: it is true from the moment the save starts
  // until the write settles, so the card keeps no second flag for it.
  const saving = shell.saving;
  // "The maximum retry delay is at least the initial one" is a relation between
  // two fields, which a per-field spec cannot state, so the page keeps it.
  const retryMax = (fields["autoMemoryRetryMaxMs"]?.text ?? "").trim();
  const retryBase = (fields["autoMemoryRetryMs"]?.text ?? "").trim();
  const inconsistent =
    retryMax !== "" && retryBase !== "" && Number(retryMax) < Number(retryBase);
  const blocked = shell.invalid || inconsistent;
  const edit = (name: string, text: string): void => {
    setSaved(false);
    form.actions.edit(name, text);
  };
  const save = async (): Promise<void> => {
    const current = form.store.getSnapshot().shell;
    if (!current.writable || !current.dirty || blocked || current.saving)
      return;
    setSaved(false);
    await form.model.save();
    // The model published the settled state before its promise resolved, so the
    // store already carries the outcome of the write.
    const after = form.store.getSnapshot().shell;
    setSaved(!after.failed && !after.dirty);
  };
  const reset = (): void => {
    setSaved(false);
    for (const spec of SETTINGS_SPECS) form.actions.resetField(spec.field);
  };
  const discard = (): void => {
    setSaved(false);
    form.actions.discard();
  };

  return (
    <div className="reme-settings-card">
      <div className="reme-settings-body">
        <div className="reme-settings-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            id="reme-tab-memory"
            aria-controls={
              tab === "memory" ? "reme-tabpanel-memory" : undefined
            }
            aria-selected={tab === "memory"}
            className={tab === "memory" ? "active" : ""}
            onClick={() => setTab("memory")}
          >
            {t("tabMemory")}
          </button>
          <button
            type="button"
            role="tab"
            id="reme-tab-environment"
            aria-controls={
              tab === "environment" ? "reme-tabpanel-environment" : undefined
            }
            aria-selected={tab === "environment"}
            className={tab === "environment" ? "active" : ""}
            onClick={() => setTab("environment")}
          >
            {t("tabEnvironment")}
          </button>
        </div>
        <div
          role="tabpanel"
          id={`reme-tabpanel-${tab}`}
          aria-labelledby={`reme-tab-${tab}`}
        >
          {tab === "environment" ? (
            rpc && envT ? (
              <ReMeEnvironmentTab scope={scope} rpc={rpc} t={envT} />
            ) : (
              <p className="reme-settings-muted">{t("loading")}</p>
            )
          ) : state.status === "ready" ? (
            <>
              {shell.dirty ? (
                <p className="reme-settings-pending">{t("unsaved")}</p>
              ) : null}
              <fieldset
                className="reme-settings-fields"
                disabled={saving || !shell.writable}
              >
                <SettingsForm fields={fields} edit={edit} t={t} />
              </fieldset>
              {!shell.writable ? (
                <p className="reme-settings-error" role="status">
                  {t("readOnly")}
                </p>
              ) : null}
              {blocked ? (
                <p className="reme-settings-error" role="status">
                  {t("invalid")}
                </p>
              ) : null}
              <div className="reme-settings-actions">
                <button
                  type="button"
                  className="reme-settings-reset"
                  disabled={!shell.writable || saving}
                  onClick={reset}
                >
                  {t("reset")}
                </button>
                <span className="reme-settings-result" role="status">
                  {shell.failed ? (
                    <span className="reme-settings-error">
                      {t("saveFailed")}
                    </span>
                  ) : null}
                  {saved && !shell.dirty ? (
                    <span className="reme-settings-success">{t("saved")}</span>
                  ) : null}
                </span>
                <button
                  type="button"
                  className="reme-settings-discard"
                  disabled={!shell.dirty || saving}
                  onClick={discard}
                >
                  {t("discard")}
                </button>
                <button
                  type="button"
                  className="reme-settings-save"
                  disabled={
                    !shell.writable || !shell.dirty || blocked || saving
                  }
                  onClick={() => void save()}
                >
                  {t(saving ? "saving" : "save")}
                </button>
              </div>
            </>
          ) : (
            <p className="reme-settings-muted">{t("loading")}</p>
          )}
        </div>
      </div>
    </div>
  );
}

/** The memory-settings controls, all reading and writing through the model. */
function SettingsForm({
  fields,
  edit,
  t,
}: {
  fields: Record<string, FieldState>;
  edit: (name: string, text: string) => void;
  t: Translator;
}): JSX.Element {
  const text = (name: string): string => fields[name]?.text ?? "";
  const invalid = (name: string): boolean => fields[name]?.invalid ?? false;
  const checked = (name: string): boolean => text(name) === "true";
  return (
    <>
      <section className="reme-settings-section">
        <h4>{t("connection")}</h4>
        <div className="reme-settings-grid">
          <Field wide label={t("endpoint")} hint={t("endpointHint")}>
            <input
              value={text("endpoint")}
              aria-invalid={invalid("endpoint")}
              onChange={(event) => edit("endpoint", event.target.value)}
            />
          </Field>
          <Field label={t("language")}>
            <select
              value={text("language")}
              onChange={(event) => edit("language", event.target.value)}
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
              value={text("searchLimit")}
              onChange={(event) => edit("searchLimit", event.target.value)}
            />
          </Field>
          <Field label={t("requestTimeout")}>
            <input
              type="number"
              min="1000"
              max="120000"
              value={text("requestTimeoutMs")}
              onChange={(event) => edit("requestTimeoutMs", event.target.value)}
            />
          </Field>
        </div>
      </section>
      <section className="reme-settings-section">
        <h4>{t("autoMemory")}</h4>
        <div className="reme-settings-grid">
          <Check
            label={t("autoMemoryEnabled")}
            checked={checked("autoMemoryEnabled")}
            onChange={(next) =>
              edit("autoMemoryEnabled", next ? "true" : "false")
            }
          />
          <Check
            label={t("rootOnly")}
            checked={checked("rootAgentsOnly")}
            onChange={(next) => edit("rootAgentsOnly", next ? "true" : "false")}
          />
          <Field label={t("autoMemoryInterval")}>
            <input
              type="number"
              min="1"
              max="1000"
              value={text("autoMemoryInterval")}
              onChange={(event) =>
                edit("autoMemoryInterval", event.target.value)
              }
            />
          </Field>
          <Field label={t("timezone")}>
            <input
              value={text("timezone")}
              aria-invalid={invalid("timezone")}
              onChange={(event) => edit("timezone", event.target.value)}
            />
          </Field>
        </div>
      </section>
      <section className="reme-settings-section">
        <h4>{t("autoDream")}</h4>
        <div className="reme-settings-grid">
          <Check
            label={t("autoDreamEnabled")}
            checked={checked("autoDreamEnabled")}
            onChange={(next) =>
              edit("autoDreamEnabled", next ? "true" : "false")
            }
          />
          <Field label={t("dreamTime")}>
            <input
              type="time"
              value={text("dreamCron")}
              onChange={(event) => edit("dreamCron", event.target.value)}
            />
          </Field>
          <Field wide label={t("dreamHint")}>
            <textarea
              value={text("dreamHint")}
              onChange={(event) => edit("dreamHint", event.target.value)}
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
              value={text("backgroundTimeoutMs")}
              onChange={(event) =>
                edit("backgroundTimeoutMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("retryDelay")}>
            <input
              type="number"
              min="1000"
              max="60000"
              value={text("autoMemoryRetryMs")}
              onChange={(event) =>
                edit("autoMemoryRetryMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("retryMaxDelay")}>
            <input
              type="number"
              min="1000"
              max="3600000"
              value={text("autoMemoryRetryMaxMs")}
              onChange={(event) =>
                edit("autoMemoryRetryMaxMs", event.target.value)
              }
            />
          </Field>
          <Field label={t("shutdownTimeout")}>
            <input
              type="number"
              min="100"
              max="60000"
              value={text("shutdownTimeoutMs")}
              onChange={(event) =>
                edit("shutdownTimeoutMs", event.target.value)
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

function installStyles(): () => void {
  const tag = document.createElement("style");
  tag.dataset.pluginCss = "@agentscope-ai/reme-dsh-plugin/settings";
  tag.textContent = `${styles}\n${environmentStyles}`;
  document.head.appendChild(tag);
  return () => tag.remove();
}
