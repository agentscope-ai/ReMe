import { createAppTransport } from "@openai/mcp-extensions/app/transport";

const $ = (id) => document.getElementById(id);
const transport = createAppTransport();
let connected = false;
let disposed = false;
let language = "en";
let zone = "UTC";
let hasSnapshot = false;
let refreshing = false;
const words = {
  en: {
    overview: "Overview", autoMemory: "Auto Memory", consolidation: "Consolidation", components: "Components",
    service: "ReMe service", queuedTurns: "Queued turns", queuedSessions: "Queued sessions",
    connectionDetails: "Connection details", viewHelp: "About this view",
    interval: "Batch interval", timezone: "Timezone", lastResult: "Last result",
    healthy: "Healthy", unhealthy: "Unhealthy", unknown: "Unknown", processMemory: "Process memory (RSS)",
    componentMemory: "Estimated component memory", noComponents: "No component memory details returned.",
    componentHelp: "Memory estimates from ReMe status. Component-level health is not included in the MCP response.",
    dreamHelp: "The daily schedule runs while the ReMe MCP connection is alive.",
    connection: "Connection", memory: "Automatic memory", recall: "Automatic recall", schedule: "Schedule",
    next: "Next run", scheduler: "Scheduler", last: "Last consolidation", activity: "Recent activity",
    details: "Service details", on: "Enabled", off: "Disabled", connected: "Connected", offline: "Unavailable",
    refresh: "Refresh", refreshing: "Refreshing…", checked: "Checked", empty: "No activity recorded yet.",
    waiting: "Waiting for the first status result…", unavailable: "Status could not be refreshed. Try again or reopen ReMe status.",
    noHost: "Open ReMe status from Codex MCP settings to view live data.",
    trust: "Connection health does not establish Hook trust. Review ReMe in Codex Hooks settings.",
    saved: "This view uses saved settings. Save changes in MCP settings, then refresh here.",
    pending: (n) => `${n} queued turns`, batch: (n, s) => `Batch size: ${n} turns · ${s} queued sessions`,
    other: (n) => `${n} turns are queued for other service addresses.`,
    noRecall: "No recall activity yet", noDream: "No consolidation recorded yet", paused: "No run scheduled",
    running: "Running", starting: "Starting", updating: "Applying settings", stopped: "Stopped",
    scheduled: "Scheduled", manual: "Manual", completed: "Completed", failed: "Failed", cancelled: "Cancelled", interrupted: "Interrupted",
    recall_found: "Relevant memory recalled", recall_empty: "No matching memory", recall_failed: "Recall failed", recall_disabled: "Recall disabled",
    memory_queued: "Turn queued", memory_saved: "Memory saved", memory_failed: "Memory delivery failed", memory_disabled: "Recording disabled",
    retry_failed: "Delivery retry failed", dream_started: "Consolidation started", dream_completed: "Consolidation completed",
    dream_failed: "Consolidation failed", dream_cancelled: "Consolidation cancelled", scheduled_dream_failed: "Scheduled consolidation failed",
    search_found: "Search found memory", search_empty: "Search found no memory", search_failed: "Search failed",
    capture_skipped: "No completed turn captured", SessionStart: "Session started", SessionEnd: "Session ended",
    Stop: "Turn completed", SubagentStop: "Subagent turn completed", UserPromptSubmit: "Prompt submitted",
  },
  zh: {
    overview: "总览", autoMemory: "自动记忆", consolidation: "记忆整理", components: "组件",
    service: "ReMe 服务", queuedTurns: "待提交轮数", queuedSessions: "待提交会话",
    connectionDetails: "连接详情", viewHelp: "使用说明",
    interval: "批次轮数", timezone: "时区", lastResult: "最近结果",
    healthy: "健康", unhealthy: "异常", unknown: "未知", processMemory: "进程内存（RSS）",
    componentMemory: "组件内存估算", noComponents: "暂无组件内存信息。",
    componentHelp: "内存为 ReMe status 返回的估算值；MCP 响应不包含组件级健康信息。",
    dreamHelp: "只有 ReMe MCP 连接存活时，每日计划才会触发。",
    connection: "服务连接", memory: "自动记录", recall: "自动召回", schedule: "整理计划", next: "下次运行",
    scheduler: "调度状态", last: "最近整理", activity: "最近活动", details: "服务详情", on: "已开启", off: "已关闭",
    connected: "连接正常", offline: "无法连接", refresh: "刷新", refreshing: "刷新中…", checked: "检查时间",
    empty: "暂无活动记录。", waiting: "正在等待首次状态结果…", unavailable: "未能刷新状态，请重试或重新打开 ReMe status。",
    noHost: "请从 Codex 的 MCP 设置中打开 ReMe status 查看实时数据。",
    trust: "服务连接正常不代表 Hook 已获信任，请在 Codex Hooks 设置中审核 ReMe 条目。",
    saved: "此页面使用已保存的配置。修改 MCP 设置后，请先保存，再刷新此页面。",
    pending: (n) => `${n} 轮待提交`, batch: (n, s) => `每批 ${n} 轮 · ${s} 个会话待提交`,
    other: (n) => `另有 ${n} 轮等待提交至其他服务地址。`, noRecall: "暂无召回活动", noDream: "暂无整理记录", paused: "暂无计划",
    running: "运行中", starting: "启动中", updating: "正在应用配置", stopped: "已停止",
    scheduled: "定时", manual: "手动", completed: "已完成", failed: "失败", cancelled: "已取消", interrupted: "已中断",
    recall_found: "已召回相关记忆", recall_empty: "未找到相关记忆", recall_failed: "召回失败", recall_disabled: "自动召回已关闭",
    memory_queued: "已加入记录队列", memory_saved: "记忆已保存", memory_failed: "记忆提交失败", memory_disabled: "自动记录已关闭",
    retry_failed: "重试提交失败", dream_started: "开始整理", dream_completed: "整理已完成", dream_failed: "整理失败",
    dream_cancelled: "整理已取消", scheduled_dream_failed: "定时整理失败", search_found: "检索到相关记忆",
    search_empty: "未检索到相关记忆", search_failed: "检索失败", capture_skipped: "未捕获到完成轮次",
    SessionStart: "会话开始", SessionEnd: "会话结束", Stop: "轮次完成", SubagentStop: "子代理轮次完成", UserPromptSubmit: "已提交提问",
  },
};
const t = (key) => words[language][key] ?? key;
const text = (id, value) => { $(id).textContent = value; };
function formatTime(value) {
  if (value == null) return "—";
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  if (!Number.isFinite(date.getTime())) return "—";
  return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en-GB", {
    timeZone: zone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", timeZoneName: "short",
  }).format(date);
}
function pill(id, value, tone = "good") { text(id, value); $(id).dataset.tone = tone; }
function toggle(id, enabled) { pill(id, t(enabled ? "on" : "off"), enabled ? "good" : "neutral"); }
function applyHost(context = {}) {
  if (["dark", "light"].includes(context.theme)) document.documentElement.dataset.theme = context.theme;
  for (const key of ["--color-background-primary", "--color-text-primary"]) {
    const value = context.styles?.variables?.[key];
    if (typeof value === "string") document.documentElement.style.setProperty(key, value);
  }
}
function fail() {
  if (disposed) return;
  text("error", t("unavailable")); $("error").hidden = false;
  $("loading").hidden = true;
  document.querySelector("main").setAttribute("aria-busy", "false");
}
function render(result) {
  if (disposed) return;
  const data = result?.structuredContent;
  if (result?.isError || !data?.service?.health_check || !data?.auto_memory || !data?.auto_dream) { fail(); return; }
  language = data.language === "zh" ? "zh" : "en";
  zone = data.auto_dream.timezone;
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  text("refresh", t(refreshing ? "refreshing" : "refresh"));
  const health = data.service.health_check;
  pill("connection-state", t(health.reachable ? "connected" : "offline"), health.reachable ? "good" : "bad");
  $("connection-state").hidden = false;
  const summary = typeof health.answer === "string" ? health.answer.trim().match(/(?:^|-\s*)(unhealthy|healthy)$/) : null;
  text("service-health", t(!health.reachable ? "offline" : summary?.[1] ?? "unknown"));
  $("service-health").dataset.tone = !health.reachable || summary?.[1] === "unhealthy" ? "bad" : summary ? "good" : "neutral";
  text("overview-queue", data.auto_memory.queued_turns);
  text("overview-sessions", data.auto_memory.queued_sessions);
  text("health", health.reachable ? (typeof health.answer === "string" ? health.answer : JSON.stringify(health.answer)) : health.error);
  text("endpoint", data.mcpUrl);
  text("plugin-version", `ReMe for Codex · ${data.plugin_version}`);
  text("checked", `${t("checked")} ${formatTime(data.checked_at)}`);
  toggle("memory-state", data.auto_memory.enabled);
  text("queue", data.auto_memory.queued_turns);
  text("interval", data.auto_memory.interval);
  text("sessions", data.auto_memory.queued_sessions);
  $("other-queue").hidden = !data.auto_memory.other_endpoint_queued_turns;
  text("other-queue", t("other")(data.auto_memory.other_endpoint_queued_turns));
  toggle("recall-state", data.auto_recall.enabled);
  const activity = data.recent_activity ?? [];
  const recall = activity.findLast((row) => row.event?.startsWith("recall_"));
  text("recall-result", recall ? t(recall.event) : t("noRecall"));
  text("recall-time", recall ? formatTime(recall.time) : "—");
  const dream = data.auto_dream;
  toggle("dream-state", dream.enabled);
  text("schedule", dream.cron);
  text("timezone", dream.timezone);
  text("next", dream.next_run_at ? formatTime(dream.next_run_at) : t("paused"));
  text("scheduler", `${t("scheduler")}: ${t(dream.phase)}`);
  const last = dream.last_run;
  text("last-result", last ? t(last.status) : "—");
  text("last", last ? `${t(last.origin)} · ${t(last.status)} · ${formatTime(last.completed_at ?? last.started_at)}${last.error ? ` · ${last.error}` : ""}` : t("noDream"));
  $("activity").replaceChildren();
  for (const row of activity.slice(-20).reverse()) {
    const item = document.createElement("li");
    const description = document.createElement("span"); description.textContent = t(row.event);
    const code = document.createElement("code"); code.textContent = `${row.event}${row.turns ? ` · ${row.turns}` : ""}${row.error ? ` · ${row.error}` : ""}`;
    description.append(code);
    const time = document.createElement("time"); time.textContent = formatTime(row.time);
    item.append(description, time); $("activity").append(item);
  }
  $("no-activity").hidden = activity.length > 0; text("no-activity", t("empty"));
  renderComponents(data.service.status);
  text("details", JSON.stringify(data.service, null, 2));
  $("error").hidden = true; $("loading").hidden = true; $("overview").hidden = false;
  document.querySelector("main").setAttribute("aria-busy", "false"); hasSnapshot = true;
}

function renderComponents(status) {
  // ReMe's MCP contract returns the human-readable answer, not HTTP response metadata.
  const report = status?.reachable && typeof status.answer === "string" ? status.answer : "";
  const amount = "([0-9]+(?:\\.[0-9]+)? (?:B|KiB|MiB|GiB|TiB))";
  text("rss", report.match(new RegExp(`^  Process RSS +${amount}$`, "m"))?.[1] ?? "—");
  text("component-total", report.match(new RegExp(`^  Components total +${amount}$`, "m"))?.[1] ?? "—");
  $("components").replaceChildren();
  for (const row of report.matchAll(new RegExp(`^  ([a-z_]+):([^\\s:]+)  +${amount}$`, "gm"))) {
    const card = document.createElement("div"); card.className = "component";
    const title = document.createElement("strong"); title.textContent = `${row[1].replaceAll("_", " ")} · ${row[2]}`;
    const size = document.createElement("div"); size.className = "value compact"; size.textContent = row[3];
    card.append(title, size); $("components").append(card);
  }
  $("no-components").hidden = $("components").childElementCount > 0;
  text("no-components", status?.reachable === false ? status.error : t("noComponents"));
}
const tabs = [...document.querySelectorAll('[role="tab"]')];
function selectTab(selected) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
    $(tab.getAttribute("aria-controls")).hidden = !active;
  }
}
for (const tab of tabs) {
  tab.addEventListener("click", () => selectTab(tab));
  tab.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(tab);
    const next = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
    if (next == null) return;
    event.preventDefault(); selectTab(tabs[next]); tabs[next].focus();
  });
}

let resizeFrame;
const resizeObserver = new ResizeObserver(() => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {
    if (connected && !disposed) transport.notify("ui/notifications/size-changed", { height: Math.ceil(document.body.getBoundingClientRect().height) });
  });
});
resizeObserver.observe(document.body);
transport.on("ui/notifications/tool-result", (result) => { try { render(result); } catch { fail(); } });
transport.on("ui/notifications/tool-cancelled", fail);
transport.on("ui/notifications/host-context-changed", applyHost);
transport.handle("ui/resource-teardown", () => {
  disposed = true; connected = false; $("refresh").disabled = true;
  resizeObserver.disconnect(); cancelAnimationFrame(resizeFrame);
  setTimeout(() => transport.dispose(), 0);
  return {};
});
$("refresh").addEventListener("click", async () => {
  if (!connected || refreshing) return;
  refreshing = true; $("refresh").disabled = true; text("refresh", t("refreshing"));
  document.querySelector("main").setAttribute("aria-busy", "true");
  try { render(await transport.request("tools/call", { name: "reme_status", arguments: {} }, 125000)); }
  catch { fail(); }
  finally { refreshing = false; $("refresh").disabled = !connected; text("refresh", t("refresh")); }
});
async function connect() {
  if (window.parent === window) { text("loading", t("noHost")); return; }
  const host = await transport.request("ui/initialize", {
    protocolVersion: "2026-01-26", appInfo: { name: "reme-status", version: "0.2.3" },
    appCapabilities: { availableDisplayModes: ["fullscreen"] },
  });
  if (host.protocolVersion !== "2026-01-26") throw new Error("Unsupported UI protocol");
  connected = true; applyHost(host.hostContext); $("refresh").disabled = false;
  if (!hasSnapshot) text("loading", t("waiting"));
  transport.notify("ui/notifications/initialized");
  const context = host.hostContext ?? {};
  if (context.displayMode !== "fullscreen" && context.availableDisplayModes?.includes("fullscreen")) {
    // One supported request only; respect the host's final placement (including a settings modal).
    await transport.request("ui/request-display-mode", { mode: "fullscreen" }).catch(() => {});
  }
}
connect().catch(fail);
