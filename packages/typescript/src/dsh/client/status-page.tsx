import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";
import { MarkdownText } from "@deepseek-ai/dsh-client-ui-primitives";

import { ReMeClient } from "../../core/client.js";
import type {
  ReMeComponentHealth,
  ReMeHealth,
  ReMeMemoryStatus,
} from "../../core/types.js";
import type { ReMeRuntimeSnapshot } from "../runtime-status.js";
import type { ReMeSettings } from "../types.js";
import { parseMarkdownFrontmatter } from "./frontmatter.js";

export const statusEn = {
  nav: "ReMe Status",
  title: "ReMe Status",
  description:
    "Monitor memory capture, consolidation, service health, and your ReMe workspace.",
  workspace: "Your memory workspace",
  overview: "Overview",
  autoMemory: "Auto Memory",
  autoDream: "Memory Consolidation",
  components: "Components",
  journal: "Journal",
  knowledge: "Personal Knowledge Base",
  website: "ReMe website",
  openReMe: "Open ReMe",
  runDream: "Consolidate Memory Now",
  runningDream: "Consolidating…",
  loading: "Loading ReMe status…",
  unavailable: "ReMe status is unavailable.",
  connected: "Connected",
  disconnected: "Unavailable",
  healthy: "Healthy",
  unhealthy: "Needs attention",
  enabled: "Enabled",
  disabled: "Disabled",
  running: "Running",
  idle: "Idle",
  stopping: "Stopping",
  version: "Version",
  endpoint: "Service URL",
  lastRefresh: "Last refreshed",
  serviceHealth: "Service health",
  processMemory: "Process memory",
  componentMemory: "Component memory",
  activeSessions: "Active sessions",
  queuedTurns: "Queued turns",
  runningTasks: "Running tasks",
  queuedTasks: "Queued tasks",
  everyTurns: "Submission interval",
  turnsUnit: "completed turns",
  turnUnit: "completed turn",
  recentTasks: "Recent submissions",
  capturePipeline: "Capture pipeline",
  conversationTurns: "Conversation turns",
  submissionQueue: "Submission queue",
  longTermMemory: "Long-term memory",
  pipelineReady: "Ready for completed conversations",
  activity: "Activity",
  noTasks: "No automatic-memory submissions in this process yet.",
  taskQueued: "Queued",
  taskRunning: "Running",
  taskCompleted: "Completed",
  taskFailed: "Failed",
  taskCancelled: "Cancelled",
  messages: "messages",
  turns: "turns",
  lastError: "Latest error",
  schedule: "Schedule",
  consolidationFlow: "Consolidation flow",
  journalSource: "Journal entries",
  organizeKnowledge: "Organize and connect",
  knowledgeOutput: "Personal knowledge base",
  nextConsolidation: "Next consolidation",
  timezone: "Timezone",
  nextRun: "Next run",
  lastRun: "Last run",
  never: "Not run in this process",
  noSchedule: "No run scheduled",
  dreamCompleted: "Memory consolidation completed.",
  dreamFailed: "Memory consolidation failed",
  componentDetails: "Component health and resource usage",
  healthyComponents: "healthy components",
  infrastructure: "Memory infrastructure",
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
  fileListLoading: "Loading files…",
  fileListEmpty: "No files yet.",
  fileListLimited: "Showing the newest 5,000 files.",
  fileCount: "files",
  fileUnit: "file",
  searchFiles: "Search files…",
  noSearchResults: "No matching files.",
  documentMetadata: "Document metadata",
  documentContent: "Document content",
  selectFile: "Select a file to preview its contents.",
  fileLoadFailed: "Could not load this file.",
  journalDescription: "Daily notes captured from conversations and sources.",
  knowledgeDescription:
    "Consolidated knowledge organized for long-term recall.",
};

export const statusZh: typeof statusEn = {
  nav: "ReMe 状态",
  title: "ReMe 状态",
  description: "查看记忆写入、知识整理、服务健康和 ReMe 工作区。",
  workspace: "你的记忆工作区",
  overview: "总览",
  autoMemory: "自动记忆",
  autoDream: "记忆整理",
  components: "组件",
  journal: "日记",
  knowledge: "个人知识库",
  website: "ReMe 官网",
  openReMe: "打开 ReMe",
  runDream: "立即整理",
  runningDream: "正在整理…",
  loading: "正在加载 ReMe 状态…",
  unavailable: "ReMe 状态不可用。",
  connected: "已连接",
  disconnected: "连接失败",
  healthy: "健康",
  unhealthy: "需要处理",
  enabled: "已启用",
  disabled: "未启用",
  running: "运行中",
  idle: "空闲",
  stopping: "正在停止",
  version: "版本",
  endpoint: "服务地址",
  lastRefresh: "最近刷新",
  serviceHealth: "服务健康",
  processMemory: "进程内存",
  componentMemory: "组件内存",
  activeSessions: "活跃会话",
  queuedTurns: "待处理回合",
  runningTasks: "运行中任务",
  queuedTasks: "排队任务",
  everyTurns: "提交间隔",
  turnsUnit: "个完成回合",
  turnUnit: "个完成回合",
  recentTasks: "最近提交",
  capturePipeline: "记忆写入流程",
  conversationTurns: "对话回合",
  submissionQueue: "提交队列",
  longTermMemory: "长期记忆",
  pipelineReady: "等待已完成的对话",
  activity: "运行记录",
  noTasks: "本次 DSH 运行期间还没有自动记忆提交。",
  taskQueued: "排队中",
  taskRunning: "运行中",
  taskCompleted: "已完成",
  taskFailed: "失败",
  taskCancelled: "已取消",
  messages: "条消息",
  turns: "个回合",
  lastError: "最近错误",
  schedule: "执行计划",
  consolidationFlow: "记忆整理流程",
  journalSource: "日记记录",
  organizeKnowledge: "整理与关联",
  knowledgeOutput: "个人知识库",
  nextConsolidation: "下次整理",
  timezone: "时区",
  nextRun: "下次执行",
  lastRun: "最近执行",
  never: "本次运行期间尚未执行",
  noSchedule: "当前没有执行计划",
  dreamCompleted: "记忆整理已完成。",
  dreamFailed: "记忆整理失败",
  componentDetails: "组件健康与资源占用",
  healthyComponents: "个健康组件",
  infrastructure: "记忆基础设施",
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
  fileListLoading: "正在加载文件…",
  fileListEmpty: "还没有文件。",
  fileListLimited: "当前显示最近的 5,000 个文件。",
  fileCount: "个文件",
  fileUnit: "个文件",
  searchFiles: "搜索文件…",
  noSearchResults: "没有匹配的文件。",
  documentMetadata: "文档元数据",
  documentContent: "文档内容",
  selectFile: "选择一个文件查看内容。",
  fileLoadFailed: "无法读取这个文件。",
  journalDescription: "从对话和其他来源自动沉淀的每日记录。",
  knowledgeDescription: "经过整理、适合长期检索和持续积累的个人知识。",
};

export type StatusTranslator = (key: keyof typeof statusEn) => string;

interface SettingsSnapshot<T> {
  status: "loading" | "ready" | "unavailable";
  value: T | undefined;
}

export interface StatusSettingsScope<T> {
  getSnapshot(): SettingsSnapshot<T>;
  subscribe(listener: () => void): () => void;
}

export interface StatusRpc {
  call(
    channel: string,
    endpoint: string,
    payload: unknown,
    signal?: AbortSignal,
  ): Promise<
    | { ok: true; value?: unknown }
    | { ok: false; error: { code: string; message: string } }
  >;
}

interface StatusData {
  loading: boolean;
  error: string;
  refreshedAt?: string;
  health?: ReMeHealth;
  memory?: ReMeMemoryStatus;
  runtime?: ReMeRuntimeSnapshot;
  appConfig?: Record<string, unknown>;
  dreamAction?: string;
}

type StatusTab =
  | "overview"
  | "auto-memory"
  | "auto-dream"
  | "components"
  | "journal"
  | "knowledge";

export interface ReMeStatusPageProps {
  scope: StatusSettingsScope<ReMeSettings>;
  rpc: StatusRpc;
  t: StatusTranslator;
}

/** Render the independent ReMe monitoring and workspace page. */
export function ReMeStatusPage({
  scope,
  rpc,
  t,
}: ReMeStatusPageProps): JSX.Element | null {
  const snapshot = useSyncExternalStore(
    (listener) => scope.subscribe(listener),
    () => scope.getSnapshot(),
  );
  const settings = snapshot.value;
  const [tab, setTab] = useState<StatusTab>("overview");
  const [data, setData] = useState<StatusData>({ loading: true, error: "" });
  const [runningDream, setRunningDream] = useState(false);

  const refresh = useCallback(
    async (runtimeOnly = false) => {
      if (settings === undefined) return;
      const controller = new AbortController();
      if (!runtimeOnly)
        setData((current) => ({ ...current, loading: true, error: "" }));
      try {
        const runtimePromise = loadRuntime(rpc, controller.signal);
        if (runtimeOnly) {
          const runtime = await runtimePromise;
          setData((current) => ({ ...current, runtime }));
          return;
        }
        const client = diagnosticClient(settings);
        const [health, status, config, runtime] = await Promise.all([
          client.healthCheck({ signal: controller.signal }),
          client.status({ signal: controller.signal }),
          client.appConfig({ signal: controller.signal }),
          runtimePromise,
        ]);
        const failed = [health, status, config].find((result) => !result.ok);
        setData((current) => ({
          ...current,
          loading: false,
          error: failed?.error ?? "",
          refreshedAt: new Date().toISOString(),
          health: health.health,
          memory: status.memory,
          runtime,
          appConfig: isRecord(config.answer) ? config.answer : undefined,
        }));
      } catch (error) {
        if (controller.signal.aborted) return;
        setData((current) => ({
          ...current,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        }));
      }
    },
    [rpc, settings],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (settings === undefined) return;
    const timer = window.setInterval(() => {
      if (!document.hidden) void refresh(true);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [refresh, settings]);

  if (snapshot.status === "unavailable") return null;
  if (settings === undefined) {
    return <p className="reme-status-state">{t("loading")}</p>;
  }

  const runDream = async () => {
    if (runningDream) return;
    setRunningDream(true);
    setData((current) => ({ ...current, error: "", dreamAction: undefined }));
    const result = await diagnosticClient(settings).autoDream({
      hint: settings.dreamHint,
    });
    setData((current) => ({
      ...current,
      error: result.ok ? "" : `${t("dreamFailed")}: ${result.error ?? ""}`,
      dreamAction: result.ok ? t("dreamCompleted") : undefined,
    }));
    setRunningDream(false);
    void refresh();
  };

  const tabs: Array<[StatusTab, keyof typeof statusEn]> = [
    ["overview", "overview"],
    ["auto-memory", "autoMemory"],
    ["auto-dream", "autoDream"],
    ["components", "components"],
    ["journal", "journal"],
    ["knowledge", "knowledge"],
  ];

  return (
    <section className="reme-status-page">
      <header className="reme-status-page-header">
        <div>
          <h2>{t("title")}</h2>
          <p>{t("description")}</p>
        </div>
      </header>

      <div className="reme-status-tabs" role="tablist">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={tab === id}
            className={tab === id ? "active" : ""}
            onClick={() => setTab(id)}
          >
            {t(label)}
          </button>
        ))}
      </div>

      {data.error ? (
        <div className="reme-status-notice error" role="status">
          {data.error}
        </div>
      ) : null}
      {data.dreamAction ? (
        <div className="reme-status-notice success" role="status">
          {data.dreamAction}
        </div>
      ) : null}

      <div className="reme-status-tab-panel" role="tabpanel">
        {tab === "overview" ? (
          <Overview settings={settings} data={data} t={t} />
        ) : null}
        {tab === "auto-memory" ? (
          <AutoMemory runtime={data.runtime} t={t} />
        ) : null}
        {tab === "auto-dream" ? (
          <AutoDream
            runtime={data.runtime}
            runningDream={runningDream}
            onRunDream={() => void runDream()}
            t={t}
          />
        ) : null}
        {tab === "components" ? (
          <Components health={data.health} memory={data.memory} t={t} />
        ) : null}
        {tab === "journal" ? (
          <MemoryFiles
            kind="journal"
            settings={settings}
            directory={directoryFrom(data.appConfig, "daily_dir", "daily")}
            t={t}
          />
        ) : null}
        {tab === "knowledge" ? (
          <MemoryFiles
            kind="knowledge"
            settings={settings}
            directory={directoryFrom(data.appConfig, "digest_dir", "digest")}
            t={t}
          />
        ) : null}
      </div>
    </section>
  );
}

function Overview({
  settings,
  data,
  t,
}: {
  settings: ReMeSettings;
  data: StatusData;
  t: StatusTranslator;
}): JSX.Element {
  const connected = data.health !== undefined && !data.error;
  const autoMemory = data.runtime?.autoMemory;
  const autoDream = data.runtime?.autoDream;
  return (
    <>
      <section className="reme-status-overview-hero">
        <div className="reme-status-overview-brand">
          <div className="reme-status-brand-mark" aria-hidden="true">
            Re
          </div>
          <div>
            <span className="reme-status-eyebrow">ReMe Memory</span>
            <h3>{t("workspace")}</h3>
            <p>{settings.endpoint}</p>
          </div>
        </div>
        <div className="reme-status-overview-health">
          <StatusBadge
            tone={connected && data.health?.healthy ? "ok" : "bad"}
            label={
              connected
                ? t(data.health?.healthy ? "connected" : "unhealthy")
                : t("disconnected")
            }
          />
          <dl>
            <div>
              <dt>{t("version")}</dt>
              <dd>{data.health?.version ?? "—"}</dd>
            </div>
            <div>
              <dt>{t("lastRefresh")}</dt>
              <dd>{formatDate(data.refreshedAt, "—")}</dd>
            </div>
          </dl>
        </div>
        <div className="reme-status-overview-actions">
          <a
            className="reme-status-button"
            href="https://reme.agentscope.io"
            target="_blank"
            rel="noreferrer"
          >
            {t("website")}
          </a>
          <a
            className="reme-status-button primary"
            href={settings.endpoint}
            target="_blank"
            rel="noreferrer"
          >
            {t("openReMe")}
          </a>
        </div>
      </section>
      <div className="reme-status-capability-grid">
        <article>
          <div>
            <span>{t("autoMemory")}</span>
            <strong>{t(autoMemory?.enabled ? "enabled" : "disabled")}</strong>
          </div>
          <small>
            {autoMemory?.enabled
              ? `${t("everyTurns")}: ${autoMemory.interval} ${t(
                  autoMemory.interval === 1 ? "turnUnit" : "turnsUnit",
                )}`
              : t("disabled")}
          </small>
        </article>
        <article>
          <div>
            <span>{t("autoDream")}</span>
            <strong>
              {t(
                autoDream?.running
                  ? "running"
                  : autoDream?.enabled
                  ? "enabled"
                  : "disabled",
              )}
            </strong>
          </div>
          <small>{formatDate(autoDream?.nextRunAt, t("noSchedule"))}</small>
        </article>
      </div>
      <div className="reme-status-metric-grid">
        <MetricCard
          title={t("processMemory")}
          value={data.memory?.process_rss ?? "—"}
        />
        <MetricCard
          title={t("componentMemory")}
          value={data.memory?.components_total ?? "—"}
        />
        <MetricCard
          title={t("activeSessions")}
          value={String(autoMemory?.activeSessions ?? 0)}
        />
        <MetricCard
          title={t("queuedTurns")}
          value={String(autoMemory?.queuedTurns ?? 0)}
        />
      </div>
      {data.appConfig ? (
        <details className="reme-status-details">
          <summary>{t("serverConfig")}</summary>
          <pre>{JSON.stringify(data.appConfig, null, 2)}</pre>
        </details>
      ) : null}
    </>
  );
}

function AutoMemory({
  runtime,
  t,
}: {
  runtime?: ReMeRuntimeSnapshot;
  t: StatusTranslator;
}): JSX.Element {
  const memory = runtime?.autoMemory;
  const busy =
    (memory?.tasksRunning ?? 0) > 0 || (memory?.queuedTurns ?? 0) > 0;
  return (
    <section className="reme-memory-dashboard">
      <div className="reme-memory-feature-hero capture">
        <div className="reme-memory-feature-copy">
          <span className="reme-status-eyebrow">{t("autoMemory")}</span>
          <h3>{t(busy ? "running" : "idle")}</h3>
          <p>
            {memory?.enabled
              ? `${t("everyTurns")}: ${memory.interval} ${t(
                  memory.interval === 1 ? "turnUnit" : "turnsUnit",
                )}`
              : t("disabled")}
          </p>
        </div>
        <div className={`reme-memory-orbit ${busy ? "busy" : ""}`}>
          <span />
          <strong>{memory?.queuedTurns ?? 0}</strong>
          <small>{t("queuedTurns")}</small>
        </div>
        <StatusBadge
          tone={memory?.enabled ? "ok" : "muted"}
          label={t(memory?.enabled ? "enabled" : "disabled")}
        />
      </div>

      <div className="reme-memory-stat-row">
        <MetricCard
          title={t("activeSessions")}
          value={String(memory?.activeSessions ?? 0)}
        />
        <MetricCard
          title={t("queuedTurns")}
          value={String(memory?.queuedTurns ?? 0)}
        />
        <MetricCard
          title={t("runningTasks")}
          value={String(memory?.tasksRunning ?? 0)}
        />
        <MetricCard
          title={t("queuedTasks")}
          value={String(memory?.tasksQueued ?? 0)}
        />
      </div>

      <section className="reme-memory-flow-card">
        <h3>{t("capturePipeline")}</h3>
        <div className="reme-memory-flow">
          <FlowStep index="01" label={t("conversationTurns")} />
          <FlowStep
            index="02"
            label={t("submissionQueue")}
            detail={String(memory?.tasksQueued ?? 0)}
          />
          <FlowStep index="03" label={t("longTermMemory")} />
        </div>
        <p>{t("pipelineReady")}</p>
      </section>

      {memory?.lastError ? (
        <div className="reme-status-notice error">
          <strong>{t("lastError")}</strong>
          <span>{memory.lastError}</span>
        </div>
      ) : null}
      <section className="reme-status-task-section">
        <div className="reme-status-section-heading">
          <div>
            <h3>{t("activity")}</h3>
            <p>{t("recentTasks")}</p>
          </div>
        </div>
        {memory?.recentTasks.length ? (
          <div className="reme-status-task-list">
            {memory.recentTasks.map((task) => (
              <details key={task.id}>
                <summary>
                  <StatusBadge
                    tone={
                      task.phase === "completed"
                        ? "ok"
                        : task.phase === "failed" || task.phase === "cancelled"
                        ? "bad"
                        : "progress"
                    }
                    label={t(taskPhaseKey(task.phase))}
                  />
                  <strong>
                    {formatDate(
                      task.finishedAt ?? task.startedAt ?? task.queuedAt,
                      "—",
                    )}
                  </strong>
                  <small>{`${task.turns} ${t("turns")} · ${task.messages} ${t(
                    "messages",
                  )}`}</small>
                </summary>
                <p>{task.error ?? task.result ?? "—"}</p>
              </details>
            ))}
          </div>
        ) : (
          <div className="reme-status-empty">{t("noTasks")}</div>
        )}
      </section>
    </section>
  );
}

function AutoDream({
  runtime,
  runningDream,
  onRunDream,
  t,
}: {
  runtime?: ReMeRuntimeSnapshot;
  runningDream: boolean;
  onRunDream: () => void;
  t: StatusTranslator;
}): JSX.Element {
  const dream = runtime?.autoDream;
  return (
    <section className="reme-memory-dashboard">
      <div className="reme-consolidation-hero">
        <div className="reme-consolidation-heading">
          <span className="reme-status-eyebrow">{t("autoDream")}</span>
          <h3>{t("nextConsolidation")}</h3>
          <strong>{formatDate(dream?.nextRunAt, t("noSchedule"))}</strong>
          <div>
            <StatusBadge
              tone={
                dream?.running ? "progress" : dream?.enabled ? "ok" : "muted"
              }
              label={t(
                dream?.running
                  ? "running"
                  : dream?.enabled
                  ? "enabled"
                  : "disabled",
              )}
            />
            <code>{dream?.cron ?? "—"}</code>
            <small>{dream?.timezone ?? "—"}</small>
          </div>
        </div>
        <button
          type="button"
          className="reme-consolidation-action"
          disabled={runningDream || dream?.running === true}
          onClick={onRunDream}
        >
          <span aria-hidden="true">✦</span>
          <strong>
            {t(runningDream || dream?.running ? "runningDream" : "runDream")}
          </strong>
        </button>
      </div>

      <section className="reme-memory-flow-card consolidation">
        <h3>{t("consolidationFlow")}</h3>
        <div className="reme-memory-flow">
          <FlowStep index="01" label={t("journalSource")} />
          <FlowStep index="02" label={t("organizeKnowledge")} />
          <FlowStep index="03" label={t("knowledgeOutput")} />
        </div>
      </section>

      <div className="reme-consolidation-details">
        <MetricCard title={t("schedule")} value={dream?.cron ?? "—"} />
        <MetricCard title={t("timezone")} value={dream?.timezone ?? "—"} />
        <MetricCard
          title={t("lastRun")}
          value={formatDate(
            dream?.lastFinishedAt ?? dream?.lastStartedAt,
            t("never"),
          )}
        />
      </div>
      {dream?.lastResult ? (
        <div
          className={`reme-status-notice ${
            dream.lastResult === "completed" ? "success" : "error"
          }`}
        >
          <strong>{t("lastRun")}</strong>
          <span>
            {dream.lastError ??
              t(
                dream.lastResult === "completed"
                  ? "dreamCompleted"
                  : "dreamFailed",
              )}
          </span>
        </div>
      ) : null}
    </section>
  );
}

function Components({
  health,
  memory,
  t,
}: {
  health?: ReMeHealth;
  memory?: ReMeMemoryStatus;
  t: StatusTranslator;
}): JSX.Element {
  const groups: Array<{
    group: string;
    name?: string;
    component?: ReMeComponentHealth;
  }> = [];
  for (const [group, instances] of Object.entries(health?.components ?? {})) {
    const entries = Object.entries(instances);
    if (!entries.length) groups.push({ group });
    else
      groups.push(
        ...entries.map(([name, component]) => ({ group, name, component })),
      );
  }
  const healthyCount = groups.filter(
    ({ component }) =>
      component?.is_healthy === true ||
      (component?.is_started === true && component.is_healthy !== false),
  ).length;
  const configuredCount = groups.filter(
    ({ component }) => component !== undefined,
  ).length;
  return (
    <section className="reme-status-components">
      <div className="reme-components-hero">
        <div>
          <span className="reme-status-eyebrow">{t("infrastructure")}</span>
          <h3>{t("components")}</h3>
          <p>{t("componentDetails")}</p>
        </div>
        <div className="reme-components-score">
          <strong>{healthyCount}</strong>
          <span>/ {configuredCount}</span>
          <small>{t("healthyComponents")}</small>
        </div>
        <StatusBadge
          tone={health?.healthy ? "ok" : "bad"}
          label={t(health?.healthy ? "healthy" : "unhealthy")}
        />
      </div>
      <div className="reme-status-component-grid">
        {groups.map(({ group, name, component }) => (
          <ComponentBlock
            key={`${group}:${name ?? "none"}`}
            group={group}
            name={name}
            component={component}
            memory={name ? memory?.components[group]?.[name]?.human : undefined}
            t={t}
          />
        ))}
      </div>
    </section>
  );
}

function ComponentBlock({
  group,
  name,
  component,
  memory,
  t,
}: {
  group: string;
  name?: string;
  component?: ReMeComponentHealth;
  memory?: string;
  t: StatusTranslator;
}): JSX.Element {
  const facts = component
    ? Object.entries(component).filter(
        ([key, value]) =>
          key !== "is_started" &&
          key !== "is_healthy" &&
          key !== "memory" &&
          value !== null &&
          value !== undefined,
      )
    : [];
  if (memory) facts.push(["memory", memory]);
  const ok =
    component?.is_healthy === true ||
    (component?.is_started === true && component?.is_healthy !== false);
  return (
    <article className={`reme-status-component ${ok ? "healthy" : ""}`}>
      <header>
        <div className="reme-component-identity">
          <span aria-hidden="true">{componentInitial(group)}</span>
          <div>
            <strong>{componentGroupLabel(group, t)}</strong>
            {name && name !== "default" ? (
              <small>{`${t("instance")}: ${name}`}</small>
            ) : null}
          </div>
        </div>
        <StatusBadge
          tone={component === undefined ? "muted" : ok ? "ok" : "bad"}
          label={
            component === undefined
              ? t("noInstances")
              : t(
                  ok
                    ? "healthy"
                    : component.is_started === false
                    ? "notStarted"
                    : "unhealthy",
                )
          }
        />
      </header>
      {facts.length ? (
        <dl>
          {facts.map(([key, value]) => (
            <div key={key}>
              <dt>{componentFieldLabel(key, t)}</dt>
              <dd>{formatComponentValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </article>
  );
}

function MemoryFiles({
  kind,
  settings,
  directory,
  t,
}: {
  kind: "journal" | "knowledge";
  settings: ReMeSettings;
  directory: string;
  t: StatusTranslator;
}): JSX.Element {
  const [files, setFiles] = useState<string[]>([]);
  const [limited, setLimited] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<string>();
  const [content, setContent] = useState<string>();
  const [contentLoading, setContentLoading] = useState(false);
  const [query, setQuery] = useState("");

  const openFile = useCallback(
    async (path: string, signal?: AbortSignal) => {
      setSelected(path);
      setContentLoading(true);
      setError("");
      const result = await diagnosticClient(settings).loadFile(path, {
        signal,
      });
      if (signal?.aborted) return;
      setContentLoading(false);
      if (!result.ok) {
        setContent(undefined);
        setError(result.error ?? t("fileLoadFailed"));
        return;
      }
      setContent(result.content ?? "");
    },
    [settings, t],
  );

  useEffect(() => {
    const controller = new AbortController();
    void diagnosticClient(settings)
      .listFiles(directory, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return;
        setFiles(result.files);
        setLimited(result.limited);
        setError(result.ok ? "" : result.error ?? "");
        if (result.files[0] !== undefined)
          void openFile(result.files[0], controller.signal);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [directory, openFile, settings]);

  const visibleFiles = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return normalized
      ? files.filter((path) => path.toLowerCase().includes(normalized))
      : files;
  }, [files, query]);
  const document = useMemo(
    () => parseMarkdownFrontmatter(content ?? ""),
    [content],
  );
  const relativePath = selected?.slice(directory.length).replace(/^\//, "");

  return (
    <section className="reme-status-files">
      <div className={`reme-library-hero ${kind}`}>
        <span className="reme-library-mark" aria-hidden="true">
          {kind === "journal" ? "J" : "K"}
        </span>
        <div>
          <h3>{t(kind)}</h3>
          <p>
            {t(
              kind === "journal"
                ? "journalDescription"
                : "knowledgeDescription",
            )}
          </p>
        </div>
        <div className="reme-library-count">
          <strong>{files.length}</strong>
          <span>{t(files.length === 1 ? "fileUnit" : "fileCount")}</span>
          <code>{directory}</code>
        </div>
      </div>
      {error ? <div className="reme-status-notice error">{error}</div> : null}
      <div className="reme-status-file-browser">
        <aside>
          <div className="reme-file-search">
            <input
              type="search"
              value={query}
              placeholder={t("searchFiles")}
              onChange={(event) => setQuery(event.currentTarget.value)}
            />
            <span>{visibleFiles.length}</span>
          </div>
          {loading ? (
            <div className="reme-status-empty">{t("fileListLoading")}</div>
          ) : null}
          {!loading && !files.length ? (
            <div className="reme-status-empty">{t("fileListEmpty")}</div>
          ) : null}
          {!loading && files.length > 0 && !visibleFiles.length ? (
            <div className="reme-status-empty">{t("noSearchResults")}</div>
          ) : null}
          {visibleFiles.map((path) => (
            <button
              type="button"
              key={path}
              className={selected === path ? "active" : ""}
              onClick={() => void openFile(path)}
              title={path}
            >
              <i aria-hidden="true">{kind === "journal" ? "J" : "K"}</i>
              <span>
                <strong>{fileName(path)}</strong>
                <small>{path.slice(directory.length).replace(/^\//, "")}</small>
              </span>
            </button>
          ))}
          {limited ? (
            <div className="reme-status-file-limit">{t("fileListLimited")}</div>
          ) : null}
        </aside>
        <article>
          {contentLoading ? (
            <div className="reme-status-empty">{t("loading")}</div>
          ) : null}
          {!contentLoading && content === undefined ? (
            <div className="reme-status-empty">{t("selectFile")}</div>
          ) : null}
          {!contentLoading && content !== undefined ? (
            <div className="reme-document-preview">
              <header>
                <div>
                  <span>
                    {kind === "journal" ? t("journal") : t("knowledge")}
                  </span>
                  <h3>{fileName(selected ?? "")}</h3>
                  <code>{relativePath}</code>
                </div>
              </header>
              {document.entries.length ? (
                <section className="reme-frontmatter">
                  <h4>{t("documentMetadata")}</h4>
                  <dl>
                    {document.entries.map(({ key, value }, index) => (
                      <div key={`${key}:${index}`}>
                        <dt>{key}</dt>
                        <dd>{value || "—"}</dd>
                      </div>
                    ))}
                  </dl>
                </section>
              ) : null}
              <section className="reme-document-content">
                <h4>{t("documentContent")}</h4>
                <div className="reme-status-markdown">
                  <MarkdownText text={document.body} />
                </div>
              </section>
            </div>
          ) : null}
        </article>
      </div>
    </section>
  );
}

function MetricCard({
  title,
  value,
}: {
  title: string;
  value: string;
}): JSX.Element {
  return (
    <div className="reme-status-metric-card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FlowStep({
  index,
  label,
  detail,
}: {
  index: string;
  label: string;
  detail?: string;
}): JSX.Element {
  return (
    <div className="reme-memory-flow-step">
      <span>{index}</span>
      <strong>{label}</strong>
      {detail ? <small>{detail}</small> : null}
    </div>
  );
}

function StatusBadge({
  tone,
  label,
}: {
  tone: "ok" | "bad" | "muted" | "progress";
  label: string;
}): JSX.Element {
  return (
    <span className={`reme-status-badge ${tone}`}>
      <i />
      {label}
    </span>
  );
}

async function loadRuntime(
  rpc: StatusRpc,
  signal: AbortSignal,
): Promise<ReMeRuntimeSnapshot> {
  const result = await rpc.call(
    "/api",
    "remeStatus/runtime",
    { args: {} },
    signal,
  );
  if (!result.ok) throw new Error(result.error.message || result.error.code);
  if (!isRuntimeSnapshot(result.value))
    throw new Error("Invalid ReMe runtime status response");
  return result.value;
}

function isRuntimeSnapshot(value: unknown): value is ReMeRuntimeSnapshot {
  return (
    isRecord(value) &&
    isRecord(value.autoMemory) &&
    isRecord(value.autoDream) &&
    typeof value.phase === "string"
  );
}

function diagnosticClient(settings: ReMeSettings): ReMeClient {
  return new ReMeClient({
    endpoint: settings.endpoint,
    apiKey: "",
    requestTimeoutMs: settings.requestTimeoutMs,
    backgroundTimeoutMs: settings.backgroundTimeoutMs,
  });
}

function directoryFrom(
  config: Record<string, unknown> | undefined,
  field: string,
  fallback: string,
): string {
  const value = config?.[field];
  return typeof value === "string" && value.trim()
    ? value.replace(/^\/+|\/+$/g, "")
    : fallback;
}

function taskPhaseKey(
  phase: ReMeRuntimeSnapshot["autoMemory"]["recentTasks"][number]["phase"],
): keyof typeof statusEn {
  return {
    queued: "taskQueued",
    running: "taskRunning",
    completed: "taskCompleted",
    failed: "taskFailed",
    cancelled: "taskCancelled",
  }[phase] as keyof typeof statusEn;
}

function componentGroupLabel(group: string, t: StatusTranslator): string {
  const labels: Record<string, keyof typeof statusEn> = {
    embedding_store: "embeddingStore",
    file_graph: "fileGraph",
    file_store: "fileStore",
    keyword_index: "keywordIndex",
  };
  return labels[group] === undefined ? readableName(group) : t(labels[group]);
}

function componentFieldLabel(field: string, t: StatusTranslator): string {
  const labels: Record<string, keyof typeof statusEn> = {
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
    memory: "memoryUsage",
  };
  return labels[field] === undefined ? readableName(field) : t(labels[field]);
}

function readableName(value: string): string {
  return value
    .split("_")
    .map((part) =>
      part ? `${part.charAt(0).toUpperCase()}${part.slice(1)}` : part,
    )
    .join(" ");
}

function componentInitial(group: string): string {
  return group
    .split("_")
    .map((part) => part.charAt(0).toUpperCase())
    .slice(0, 2)
    .join("");
}

function fileName(path: string): string {
  return path.split("/").at(-1) || path;
}

function formatComponentValue(value: unknown): string {
  if (typeof value === "string" || typeof value === "number")
    return String(value);
  if (typeof value === "boolean") return value ? "✓" : "—";
  return JSON.stringify(value);
}

function formatDate(value: string | undefined, fallback: string): string {
  if (!value) return fallback;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? fallback : date.toLocaleString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
