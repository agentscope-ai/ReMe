import { useContext, useEffect, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  ArrowUpRight,
  Bell,
  Boxes,
  Check,
  ChevronDown,
  ChevronRight,
  FileText,
  FolderOpen,
  Inbox,
  Plus,
  Radio,
  Send,
  Waves,
  Ellipsis,
  Search,
  Pause,
  Play,
  X,
} from "lucide-react";
import type { StudioState } from "../../../client/src/protocol.ts";
import type {
  ConnectorView,
  OverviewAgent,
  OverviewInboxItem,
  OverviewReminder,
} from "../../../client/src/wire.ts";
import { errorText, translator } from "../../../client/src/i18n.ts";
import { useHost } from "../host.ts";
import {
  useResource,
  useWorkspace,
  formatDate,
  plainText,
} from "../resources.ts";
import SourceEditor from "./source-editor.tsx";
import FlowLinks, { type Flight } from "./links.tsx";
import { flowEdge, sourceIcon, displayAgentName as agentName } from "./flow.ts";
import FlowDrawer from "./drawer.tsx";
import { Facts, AgentDetail, EventDetail, ReminderDetail } from "./details.tsx";
import {
  TaskPanel,
  TaskPicker,
  type TaskView,
  taskViews,
} from "./task-panel.tsx";
import TaskEditor from "../tasks/editor.tsx";
import AgentResources from "../agent-resources.tsx";
import WorkspaceTitlebar, { WorkspaceLayout } from "../workspace-titlebar.tsx";
import ContextMenu, {
  type MenuItem,
  type MenuPosition,
} from "../context-menu.tsx";
import "./proactive.css";

type T = ReturnType<typeof translator>;
type Selection =
  | { type: "source"; source?: ConnectorView }
  | { type: "queues" }
  | { type: "directory"; path: string }
  | { type: "event"; item: OverviewInboxItem }
  | { type: "agent"; agentName: string; view: TaskView }
  | { type: "tasks" }
  | { type: "create"; trigger: "event" | "time" }
  | { type: "mail"; agentName: string; outbox: boolean }
  | { type: "files"; title: string; paths: string[] }
  | { type: "info" }
  | { type: "reminder"; item: OverviewReminder };
function phase(agent: OverviewAgent, t: T): string {
  const phases = {
    thinking: "flowThinking",
    working: "flowWorking",
    remembering: "flowRemembering",
    answering: "flowAnswering",
    committing: "flowCommitting",
    completed: "completed",
    interrupted: "flowInterrupted",
    failed: "agentError",
    waiting: "flowExecutionWait",
  } as const;
  const waiting = {
    paused: "taskPause",
    waiting_input: "flowWaitEvent",
    waiting_schedule: "flowWaitTime",
    cooldown: "flowCooldown",
    retry_backoff: "flowRetry",
    execution_wait: "flowExecutionWait",
    stopped: "flowInterrupted",
  } as const;
  const current = agent.active_run?.phase;
  if (current && current in phases)
    return t(phases[current as keyof typeof phases]);
  if (agent.waiting_reason && agent.waiting_reason in waiting)
    return t(waiting[agent.waiting_reason as keyof typeof waiting]);
  return t(
    agent.error
      ? "agentError"
      : agent.status === "active"
        ? "agentRunning"
        : "agentIdle",
  );
}
function clock(
  value: string | null,
  language: StudioState["settings"]["language"],
) {
  const date = value ? new Date(value) : null;
  return date && Number.isFinite(date.getTime())
    ? date.toLocaleTimeString(language === "zh" ? "zh-CN" : "en", {
        hour12: false,
      })
    : "—";
}
export default function ProactiveOverview({ state }: { state: StudioState }) {
  const t = translator(state.settings.language);
  const language = state.settings.language;
  const host = useHost();
  const layout = useContext(WorkspaceLayout);
  const request = useResource();
  const active = state.activePage === "tasks";
  const packet = state.proactive;
  const overview = packet?.overview;
  const stage = useRef<HTMLDivElement>(null);
  const nodes = useRef(new Map<string, HTMLElement>());
  const bind = (key: string) => (element: HTMLDivElement | null) => {
    if (element) nodes.current.set(key, element);
    else nodes.current.delete(key);
  };
  const [selection, setSelection] = useState<Selection | null>(null);
  const [focus, setFocus] = useState("");
  const [focusSources, setFocusSources] = useState<string[]>([]);
  const [menu, setMenu] = useState<{
    position: MenuPosition;
    items: MenuItem[];
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  useEffect(() => {
    if (!active) {
      setSelection(null);
      setMenu(null);
    }
  }, [active]);
  const lastFocus = useRef(0);
  useEffect(() => {
    if (
      !active ||
      !state.focusRequest ||
      state.focusRequest === lastFocus.current ||
      !state.focusTarget
    )
      return;
    lastFocus.current = state.focusRequest;
    const name = state.selectedAgentName;
    if (!name || name === "Remy") return;
    setFocus(name);
    setFocusSources([]);
    setSelection({
      type: "agent",
      agentName: name,
      view: state.focusTarget.kind === "mail" ? "outbox" : "chat",
    });
  }, [active, state.focusRequest, state.focusTarget, state.selectedAgentName]);
  const [expanded, setExpanded] = useState(true);
  const [motion, setMotion] = useState(true);
  const [reduced, setReduced] = useState(
    () =>
      typeof matchMedia === "function" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  const [visible, setVisible] = useState(
    () => typeof document === "undefined" || !document.hidden,
  );
  const [flights, setFlights] = useState<Flight[]>([]);
  const playback = useRef({
    seen: new Set(packet?.actions.map((action) => action.id)),
    enabled: false,
  });
  const [issue, setIssue] = useState<unknown>(null);
  const iconKey = `reme:flow-icons:${JSON.stringify([state.settings.endpoint, state.clientInfo?.workspace_dir])}`;
  const [iconState, setIcons] = useState<{
    key: string;
    values: Record<string, string>;
  }>({ key: "", values: {} });
  const icons = iconState.key === iconKey ? iconState.values : {};
  useEffect(() => {
    playback.current = {
      seen: new Set(packet?.actions.map((action) => action.id)),
      enabled: false,
    };
    setFlights([]);
  }, [iconKey]);
  useEffect(() => {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem(iconKey) ?? "{}");
      setIcons({
        key: iconKey,
        values:
          saved && typeof saved === "object" && !Array.isArray(saved)
            ? Object.fromEntries(
                Object.entries(saved).filter(
                  ([, value]) =>
                    typeof value === "string" && value.length <= 16,
                ),
              )
            : {},
      });
    } catch {
      setIcons({ key: iconKey, values: {} });
    }
  }, [iconKey]);
  useEffect(() => {
    const media = matchMedia("(prefers-reduced-motion: reduce)");
    const changed = () => setReduced(media.matches);
    const visibility = () => setVisible(!document.hidden);
    media.addEventListener("change", changed);
    document.addEventListener("visibilitychange", visibility);
    return () => {
      media.removeEventListener("change", changed);
      document.removeEventListener("visibilitychange", visibility);
    };
  }, []);
  useEffect(() => {
    const recent =
      packet?.actions.filter((action) => {
        if (playback.current.seen.has(action.id)) return false;
        playback.current.seen.add(action.id);
        if (playback.current.seen.size > 128)
          playback.current.seen.delete(
            playback.current.seen.values().next().value!,
          );
        return true;
      }) ?? [];
    const enabled = Boolean(
      active && packet?.live && motion && !reduced && visible,
    );
    const resumed = enabled && !playback.current.enabled;
    playback.current.enabled = enabled;
    if (!enabled || resumed) {
      setFlights([]);
      return;
    }
    const actions = recent.flatMap((action, index) => {
      const edge = flowEdge(action);
      const delay = Math.min(index * 140, 700);
      return edge
        ? [{ action, edge, delay, expires: Date.now() + delay + 1450 }]
        : [];
    });
    if (actions.length)
      setFlights((value) =>
        [
          ...value.filter((item) => item.expires > Date.now()),
          ...actions,
        ].slice(-12),
      );
  }, [packet?.sequence, packet?.live, motion, reduced, visible, active]);
  useEffect(() => {
    if (!flights.length) return;
    const timer = setTimeout(
      () =>
        setFlights((value) =>
          value.filter((item) => item.expires > Date.now()),
        ),
      Math.max(
        0,
        Math.min(...flights.map((item) => item.expires)) - Date.now(),
      ) + 10,
    );
    return () => clearTimeout(timer);
  }, [flights]);
  const agents =
    overview?.agents.filter((agent) => agent.agent_type !== "chat") ?? [];
  const icon = (source: Pick<ConnectorView, "source_id" | "trigger_type">) =>
    icons[source.source_id] || sourceIcon(source);
  const open = (path: string) => {
    if (
      [state.clientInfo?.resource_dir, state.clientInfo?.daily_dir].includes(
        path,
      )
    ) {
      setSelection({ type: "directory", path });
      return;
    }
    setSelection(null);
    void host.action({ type: "document.open", path }).catch(setIssue);
  };
  const linked = (paths: string[]) => (
    <div className="flow-detail-files">
      {[...new Set(paths)].map((path) => (
        <button
          type="button"
          className="flow-file"
          key={path}
          onClick={() => open(path)}
        >
          <FileText size={15} />
          <span title={path}>
            <strong>{path.split("/").at(-1) || path}</strong>
            <small>
              {path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : null}
            </small>
          </span>
          <ArrowUpRight size={14} />
        </button>
      ))}
    </div>
  );
  const clearFocus = () => {
    setFocus("");
    setFocusSources([]);
  };
  const openTask = (name: string, view: TaskView = "overview") => {
    setFocus(name);
    setFocusSources([]);
    setSelection({ type: "agent", agentName: name, view });
  };
  const focusAgent = (agent: OverviewAgent) => openTask(agent.agent_name);
  const onAgent = (name: string) => openTask(name, "manage");
  const onCreateTime = () => setSelection({ type: "create", trigger: "time" });
  const onInbox = (name: string, outbox = false) => {
    if (name === "Remy")
      setSelection({ type: "mail", agentName: name, outbox });
    else openTask(name, outbox ? "outbox" : "inbox");
  };
  const taskAgent =
    selection?.type === "agent"
      ? agents.find((agent) => agent.agent_name === selection.agentName)
      : undefined;
  const manageTask = async (agent: OverviewAgent) => {
    if (busy) return;
    setBusy(true);
    setIssue(null);
    try {
      await request({
        type: "task",
        action: "update",
        agent_name: agent.agent_name,
        settings: { enabled: !agent.enabled },
      });
    } catch (error) {
      setIssue(error);
    } finally {
      setBusy(false);
    }
  };
  const showMenu = (button: HTMLButtonElement, items: MenuItem[]) => {
    const rect = button.getBoundingClientRect();
    setMenu({ position: { x: rect.right - 190, y: rect.bottom + 6 }, items });
  };
  const sourceName = (source: {
    name: string;
    source_id: string;
    trigger_type: string;
  }) => {
    if (source.trigger_type !== "time") return source.name;
    const name = overview?.sources.find(
      (item) => item.source_id === source.source_id,
    )?.runtime_names[0];
    return name
      ? agentName(name, t)
      : source.name === "time"
        ? t("flowTimeType")
        : agentName(source.name, t);
  };
  const selectEvent = (item: OverviewInboxItem) => {
    const name = item.notification?.source_agent ?? item.agent_name;
    setFocus(name === "Remy" ? "" : name);
    setFocusSources(item.source_refs.map((source) => source.source_id));
    setSelection({ type: "event", item });
  };
  const activeCount = agents.filter(
    (agent) => agent.active_run || agent.status === "active",
  ).length;
  const sourceActive = (source: ConnectorView) =>
    focusSources.length
      ? focusSources.includes(source.source_id)
      : !focus || source.runtime_names.includes(focus);
  const agentActive = (name: string) =>
    focus
      ? name === focus
      : !focusSources.length ||
        overview?.sources.some(
          (source) =>
            focusSources.includes(source.source_id) &&
            source.runtime_names.includes(name),
        );
  const focused = Boolean(focus || focusSources.length);
  const focusLabel = focus
    ? agentName(focus, t)
    : overview?.sources
        .filter((source) => focusSources.includes(source.source_id))
        .map(sourceName)
        .join(" · ");
  const connectionLabel = t(
    packet?.live ? "flowLive" : overview ? "flowOffline" : "flowConnecting",
  );
  const status = (
    <span
      className="flow-connection"
      data-live={packet?.live}
      role="status"
      aria-label={connectionLabel}
      title={`${connectionLabel} · ${t("flowUpdated", { time: clock(packet?.asOf ?? null, language) })}`}
    >
      <i aria-hidden />
    </span>
  );
  const viewMenu = (
    <button
      className="titlebar-button"
      type="button"
      aria-label={t("flowViewSettings")}
      aria-haspopup="menu"
      aria-expanded={Boolean(menu)}
      onClick={(event) =>
        showMenu(event.currentTarget, [
          {
            id: "motion",
            label: t("flowMotion"),
            icon: Waves,
            checked: motion && !reduced,
            disabled: reduced,
            select: () => setMotion((value) => !value),
          },
          {
            id: "find",
            label: t("taskSearch"),
            icon: Search,
            select: () => setSelection({ type: "tasks" }),
          },
          {
            id: "info",
            label: t("flowRuntimeInfo"),
            icon: Activity,
            select: () => setSelection({ type: "info" }),
          },
        ])
      }
    >
      <Ellipsis size={18} />
    </button>
  );
  const unsupported =
    state.clientInfo &&
    !state.clientInfo.jobs.includes("watch_proactive_overview");
  const stateLabel = (value: OverviewInboxItem["state"]) =>
    t(
      value === "pending"
        ? "flowPending"
        : value === "processing"
          ? "flowProcessing"
          : "flowHandled",
    );
  const inboxRows = (items: OverviewInboxItem[], empty: string) =>
    items.length ? (
      <div className="flow-entries">
        {items.map((item) => (
          <button
            type="button"
            className="flow-entry"
            key={`${item.agent_name}:${item.id}`}
            data-dim={Boolean(
              focus && item.agent_name !== "Remy" && item.agent_name !== focus,
            )}
            onClick={() => selectEvent(item)}
          >
            <div className="flow-entry-meta">
              <time title={formatDate(item.received_at, language)}>
                {item.received_at
                  ? clock(item.received_at, language)
                  : t("flowAdmissionUnknown")}
              </time>
              <span className="flow-state" data-state={item.state}>
                {stateLabel(item.state)}
              </span>
            </div>
            <p>
              {item.event?.description ??
                `${agentName(item.notification?.source_agent ?? item.agent_name, t)} · ${t("flowRecordReady")}`}
            </p>
            <div className="flow-entry-source">
              <span>
                {item.source_refs
                  .map((ref) => `${icon(ref)} ${sourceName(ref)}`)
                  .join(" · ") || t("flowUnknownSource")}
              </span>
              {item.source_refs.length > 0 && (
                <span
                  className="flow-type"
                  data-time={item.source_refs.some(
                    (ref) => ref.trigger_type === "time",
                  )}
                >
                  {item.source_refs.some((ref) => ref.trigger_type === "time")
                    ? "time"
                    : "event"}
                </span>
              )}
              <small>
                {agentName(
                  item.notification?.source_agent ?? item.agent_name,
                  t,
                )}
              </small>
            </div>
          </button>
        ))}
      </div>
    ) : (
      <div className="flow-empty">
        <Inbox size={22} />
        <p>{empty}</p>
      </div>
    );
  return (
    <div className="proactive-overview">
      {layout ? (
        <WorkspaceTitlebar
          {...layout}
          path={[{ label: t("proactiveAi") }]}
          status={status}
        >
          {viewMenu}
        </WorkspaceTitlebar>
      ) : (
        <header className="flow-page-title">
          <h1>{t("proactiveAi")}</h1>
          {status}
          {viewMenu}
        </header>
      )}
      {overview && !packet?.live && (
        <p className="flow-disconnected" role="status">
          {t("flowOffline")} ·{" "}
          {t("flowUpdated", { time: clock(packet?.asOf ?? null, language) })}
        </p>
      )}
      {(issue ?? packet?.error) != null && (
        <p className="action-error" role="alert">
          {errorText(issue ?? packet?.error, language)}
        </p>
      )}
      {!overview ? (
        <div className="flow-loading">
          <Radio size={25} />
          <p>{t(unsupported ? "flowUnavailable" : "flowConnecting")}</p>
        </div>
      ) : (
        <div className="flow-scroll">
          {focused && (
            <button
              type="button"
              className="flow-focus-chip"
              onClick={clearFocus}
              aria-label={t("flowClearFocus")}
            >
              <span>{focusLabel}</span>
              <X size={13} />
            </button>
          )}
          <div
            className="flow-stage"
            ref={stage}
            data-live={packet?.live}
            onClick={(event) => {
              if (
                !(event.target as HTMLElement).closest(
                  "button, a, input, select, label",
                )
              )
                clearFocus();
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") clearFocus();
            }}
          >
            <FlowLinks
              stage={stage}
              nodes={nodes}
              overview={overview}
              flights={flights}
              labels={{ read: t("flowRead"), write: t("flowWrite") }}
              activeSources={
                focused
                  ? overview.sources
                      .filter(sourceActive)
                      .map((source) => source.source_id)
                  : undefined
              }
            />
            <div className="flow-sources" ref={bind("sources")}>
              <div className="flow-column-label">
                <Radio size={14} />
                <h2>{t("flowSources")}</h2>
                <span>{overview.sources.length}</span>
                <button
                  type="button"
                  className="flow-node-action"
                  disabled={!packet?.live}
                  aria-label={t("flowAddSource")}
                  title={t("flowAddSource")}
                  onClick={() => setSelection({ type: "source" })}
                >
                  <Plus size={15} />
                </button>
              </div>
              <div className="flow-source-list" ref={bind("source-list")}>
                {overview.sources.map((source) => (
                  <div
                    ref={bind(`source:${source.source_id}`)}
                    key={source.source_id}
                    className="flow-source"
                    data-time={source.trigger_type === "time"}
                    data-dim={!sourceActive(source)}
                    data-enabled={source.enabled !== false}
                    data-arriving={flights.some(
                      (flight) => flight.edge === `source:${source.source_id}`,
                    )}
                  >
                    <button
                      type="button"
                      title={`${source.name} · ${t("flowCollectedAt")} ${formatDate(source.last_collected_at, language)}`}
                      onClick={() => {
                        setFocus("");
                        setFocusSources([source.source_id]);
                        setSelection({ type: "source", source });
                      }}
                    >
                      <span className="flow-source-icon" aria-hidden>
                        {icon(source)}
                      </span>
                      <span className="flow-source-body">
                        <strong>{sourceName(source)}</strong>
                        <span>
                          <span
                            className="flow-type"
                            data-time={source.trigger_type === "time"}
                          >
                            {source.trigger_type}
                          </span>
                          <i
                            className="flow-source-dot"
                            data-error={Boolean(
                              source.last_error ||
                              (source.connected === false && source.enabled),
                            )}
                            data-paused={
                              source.enabled !== true ||
                              source.available !== true
                            }
                          />
                        </span>
                      </span>
                    </button>
                  </div>
                ))}
                {!overview.sources.length && (
                  <p className="flow-hint">{t("flowNoSources")}</p>
                )}
              </div>
            </div>
            <div
              className="flow-node flow-inbox"
              ref={bind("inbox")}
              data-receiving={flights.some((flight) =>
                flight.edge.startsWith("source:"),
              )}
            >
              <div className="flow-node-heading">
                <span className="flow-node-icon">
                  <Inbox size={18} />
                </span>
                <div>
                  <span className="flow-step">02 / INPUT</span>
                  <h2>{t("flowEventInbox")}</h2>
                </div>
                <span className="flow-node-total">
                  {overview.event_inbox.waiting_count}
                </span>
              </div>
              <p className="flow-node-subtitle">
                {t("flowRecentInputs")}
                <span>
                  {overview.event_inbox.processing_count} {t("flowProcessing")}
                </span>
              </p>
              {inboxRows(overview.event_inbox.items, t("flowNoEvents"))}
              <button
                className="flow-node-footer"
                disabled={!agents.length}
                onClick={() =>
                  focus ? onInbox(focus) : setSelection({ type: "queues" })
                }
              >
                {t("flowViewAll")}
                <ArrowUpRight size={13} />
              </button>
            </div>
            <div
              className="flow-node flow-runtime"
              ref={bind("runtime")}
              data-receiving={flights.some(
                (flight) => flight.edge === "claim" || flight.edge === "read",
              )}
            >
              <div className="flow-node-heading">
                <span className="flow-node-icon">
                  <Boxes size={18} />
                </span>
                <div>
                  <span className="flow-step">03 / PROCESS</span>
                  <h2>{t("flowRuntime")}</h2>
                </div>
                <span className="flow-runtime-count">{agents.length}</span>
                <button
                  type="button"
                  className="flow-node-action"
                  aria-label={t("taskSearch")}
                  title={t("taskSearch")}
                  onClick={() => setSelection({ type: "tasks" })}
                >
                  <Search size={14} />
                </button>
                <button
                  type="button"
                  className="flow-node-action"
                  aria-label={t("taskCreate")}
                  title={t("taskCreate")}
                  disabled={!packet?.live}
                  onClick={() =>
                    setSelection({ type: "create", trigger: "event" })
                  }
                >
                  <Plus size={15} />
                </button>
              </div>
              <div className="flow-runtime-summary">
                <span
                  className="flow-active-count"
                  data-working={Boolean(activeCount && packet?.live)}
                >
                  <i />
                  {activeCount
                    ? `${activeCount} ${t("agentRunning")}`
                    : t("agentIdle")}
                </span>
                <button
                  type="button"
                  className="icon-button"
                  aria-label={t(expanded ? "flowHideAgents" : "flowShowAgents")}
                  aria-expanded={expanded}
                  onClick={() => setExpanded((value) => !value)}
                >
                  {expanded ? (
                    <ChevronDown size={16} />
                  ) : (
                    <ChevronRight size={16} />
                  )}
                </button>
              </div>
              {expanded && (
                <div className="flow-agent-list">
                  {agents.map((agent) => (
                    <div
                      className="flow-agent-row"
                      key={agent.agent_name}
                      data-dim={!agentActive(agent.agent_name)}
                    >
                      <button
                        type="button"
                        className="flow-agent"
                        data-working={Boolean(packet?.live && agent.active_run)}
                        data-error={Boolean(agent.error)}
                        data-focused={focus === agent.agent_name}
                        data-recent={flights.some(
                          (flight) =>
                            flight.action.agent_name === agent.agent_name,
                        )}
                        onClick={() => focusAgent(agent)}
                      >
                        <span className="flow-agent-dot" />
                        <span>
                          <strong>{agentName(agent.agent_name, t)}</strong>
                          <small>
                            {packet?.live ? phase(agent, t) : t("flowOffline")}
                          </small>
                        </span>
                        <span
                          className="flow-type"
                          data-time={agent.agent_type === "time"}
                        >
                          {agent.agent_type}
                        </span>
                      </button>
                      <button
                        type="button"
                        className="flow-agent-menu"
                        aria-label={t("taskMenu", {
                          name: agentName(agent.agent_name, t),
                        })}
                        aria-haspopup="menu"
                        onClick={(event) =>
                          showMenu(event.currentTarget, [
                            ...taskViews.map(([view, label]) => ({
                              id: view,
                              label: t(label),
                              select: () => openTask(agent.agent_name, view),
                            })),
                            {
                              id: "toggle",
                              label: t(
                                agent.enabled ? "taskPause" : "taskEnable",
                              ),
                              icon: agent.enabled ? Pause : Play,
                              disabled: busy || !packet?.live,
                              select: () => void manageTask(agent),
                            },
                            {
                              id: "directory",
                              label: t("taskOpenDirectory"),
                              icon: FolderOpen,
                              select: () => {
                                void host
                                  .action({
                                    type: "agent.reveal",
                                    agentName: agent.agent_name,
                                  })
                                  .catch(setIssue);
                              },
                            },
                          ])
                        }
                      >
                        <Ellipsis size={15} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
              <div className="flow-runtime-deliveries">
                <span>
                  <Send size={13} />
                  {t("flowAwaitingDelivery")}
                </span>
                <strong>
                  {agents.reduce(
                    (count, agent) => count + agent.pending_delivery_count,
                    0,
                  )}
                </strong>
              </div>
              <button
                className="flow-node-footer"
                disabled={!agents.length}
                onClick={() => setSelection({ type: "tasks" })}
              >
                <span>{t("flowAllAgents")}</span>
                <ArrowUpRight size={13} />
              </button>
            </div>
            <div
              className="flow-node flow-remy"
              ref={bind("remy")}
              data-receiving={flights.some(
                (flight) => flight.edge === "deliver",
              )}
            >
              <div className="flow-node-heading">
                <span className="flow-node-icon">
                  <Send size={18} />
                </span>
                <div>
                  <span className="flow-step">04 / HANDOFF</span>
                  <h2>{t("flowRemyInbox")}</h2>
                </div>
                <span className="flow-node-total">
                  {overview.remy_inbox.waiting_count}
                </span>
              </div>
              <p className="flow-node-subtitle">{t("flowRemyHint")}</p>
              {inboxRows(overview.remy_inbox.items, t("flowNoNotifications"))}
              <button
                className="flow-node-footer"
                onClick={() => onInbox("Remy")}
              >
                {t("flowViewAll")}
                <ArrowUpRight size={13} />
              </button>
            </div>
            <div
              className="flow-node flow-reminders"
              ref={bind("reminders")}
              data-receiving={flights.some(
                (flight) => flight.edge === "remind",
              )}
            >
              <div className="flow-node-heading">
                <span className="flow-node-icon">
                  <Bell size={18} />
                </span>
                <div>
                  <span className="flow-step">05 / NOTIFY</span>
                  <h2>{t("flowReminders")}</h2>
                </div>
                <span className="flow-node-total">
                  {overview.reminders.length}
                </span>
              </div>
              <p className="flow-node-subtitle">{t("flowRemindersHint")}</p>
              {overview.reminders.length ? (
                <div className="flow-entries">
                  {overview.reminders.map((item) => (
                    <button
                      className="flow-entry"
                      type="button"
                      key={item.id}
                      onClick={() => setSelection({ type: "reminder", item })}
                    >
                      <div className="flow-entry-meta">
                        <time>{clock(item.delivered_at, language)}</time>
                        <span className="flow-state" data-state="sent">
                          <Check size={10} />
                          {t("flowSent")}
                        </span>
                      </div>
                      <p>{plainText(item.summary)}</p>
                      <div className="flow-entry-source">
                        {item.source_refs.map((ref) => (
                          <span key={ref.source_id}>
                            {icon(ref)} {sourceName(ref)}
                          </span>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="flow-empty">
                  <Bell size={23} />
                  <p>{t("flowNoReminders")}</p>
                </div>
              )}
            </div>
            <div
              className="flow-workspace"
              ref={bind("workspace")}
              data-receiving={flights.some((flight) => flight.edge === "write")}
            >
              <header>
                <span className="flow-node-icon">
                  <FolderOpen size={17} />
                </span>
                <h2>{t("flowWorkspace")}</h2>
                <span>{t("flowWorkspaceHint")}</span>
              </header>
              <div className="flow-workspace-columns">
                {(
                  [
                    [["resource"], "flowResources", FolderOpen],
                    [["task", "memory"], "flowTasks", FileText],
                    [["record"], "flowRecords", Activity],
                  ] as const
                ).map(([roles, label, Icon]) => {
                  const files = overview.workspace.filter(
                    (file) =>
                      (roles as readonly string[]).includes(file.role) &&
                      (!focus ||
                        !file.agent_name ||
                        file.agent_name === "Remy" ||
                        file.agent_name === focus),
                  );
                  return (
                    <section key={label}>
                      <h3>
                        <Icon size={15} />
                        {t(label)}
                        <button
                          type="button"
                          className="flow-file-count"
                          aria-label={`${t("flowViewAll")} · ${t(label)}`}
                          onClick={() =>
                            setSelection({
                              type: "files",
                              title: t(label),
                              paths: files.map((file) => file.path),
                            })
                          }
                        >
                          {files.length}
                          <ArrowUpRight size={11} />
                        </button>
                      </h3>
                      {files.slice(0, 3).map((file) => (
                        <button
                          key={file.id}
                          className="flow-file"
                          type="button"
                          title={file.path}
                          onClick={() => open(file.path)}
                        >
                          <span>
                            {file.agent_name
                              ? `${agentName(file.agent_name, t)} / `
                              : ""}
                            {file.path.split("/").at(-1)}
                          </span>
                          <ArrowUpRight size={12} />
                        </button>
                      ))}
                      {!files.length && <p>{t("flowNoFiles")}</p>}
                    </section>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}
      {active && selection && (
        <FlowDrawer
          closeLabel={t("closeDetails")}
          expandLabel={t("flowExpandDetails")}
          collapseLabel={t("flowCollapseDetails")}
          wide={
            selection.type === "agent" ||
            selection.type === "create" ||
            selection.type === "mail"
          }
          title={
            selection.type === "source"
              ? selection.source?.trigger_type === "time"
                ? sourceName(selection.source)
                : t(selection.source ? "flowSourceSettings" : "flowAddSource")
              : selection.type === "directory"
                ? selection.path
                : selection.type === "queues"
                  ? t("flowEventInbox")
                  : selection.type === "agent"
                    ? agentName(selection.agentName, t)
                    : selection.type === "create"
                      ? t("taskCreate")
                      : selection.type === "tasks"
                        ? t("flowAllAgents")
                        : selection.type === "files"
                          ? selection.title
                          : selection.type === "mail"
                            ? t("flowRemyInbox")
                            : selection.type === "info"
                              ? t("flowRuntimeInfo")
                              : selection.type === "event"
                                ? t(
                                    selection.item.event
                                      ? "flowEventInbox"
                                      : "flowRemyInbox",
                                  )
                                : t("flowReminders")
          }
          onClose={() => setSelection(null)}
        >
          {selection.type === "tasks" && (
            <TaskPicker state={state} agents={agents} onSelect={openTask} />
          )}
          {selection.type === "files" && linked(selection.paths)}
          {selection.type === "create" && (
            <TaskEditor
              state={state}
              initialTrigger={selection.trigger}
              onCreated={(name) => openTask(name, "manage")}
            />
          )}
          {selection.type === "mail" && (
            <AgentResources
              state={state}
              agentName={selection.agentName}
              agentLabel={(name) => agentName(name, t)}
              active={active}
              embedded
              inbox={!selection.outbox}
            />
          )}
          {selection.type === "info" && (
            <>
              <Facts
                values={[
                  [t("flowStatus"), connectionLabel],
                  [t("flowLastUpdated"), formatDate(packet?.asOf, language)],
                ]}
              />
              {overview &&
                Object.entries(overview.errors).map(([owner, error]) => (
                  <p className="action-error" key={owner}>
                    <strong>{owner}</strong> {error}
                  </p>
                ))}
            </>
          )}
          {selection.type === "directory" && (
            <DirectoryDetail
              path={selection.path}
              state={state}
              linked={linked}
            />
          )}
          {selection.type === "queues" &&
            agents.map((agent) => {
              const counts = overview?.event_inbox.queues[agent.agent_name];
              return (
                <button
                  className="flow-file"
                  type="button"
                  key={agent.agent_name}
                  onClick={() => {
                    setSelection(null);
                    onInbox(agent.agent_name);
                  }}
                >
                  <span>
                    {agentName(agent.agent_name, t)}
                    <small>
                      {t("flowPending")} {counts?.waiting_count ?? 0} ·{" "}
                      {t("flowProcessing")} {counts?.processing_count ?? 0}
                    </small>
                  </span>
                  <ArrowUpRight size={14} />
                </button>
              );
            })}
          {selection.type === "source" && (
            <SourceEditor
              key={selection.source?.source_id ?? "new"}
              state={state}
              source={selection.source}
              agents={agents}
              icon={selection.source ? icon(selection.source) : undefined}
              onSave={(id, value) => {
                const next = { ...icons };
                if (value) next[id] = value;
                else delete next[id];
                setIcons({ key: iconKey, values: next });
                try {
                  localStorage.setItem(iconKey, JSON.stringify(next));
                } catch {
                  /* Display preferences remain available for this page. */
                }
                setSelection(null);
              }}
              onTask={(name) => {
                setSelection(null);
                onAgent(name);
              }}
              onCreateTime={() => {
                setSelection(null);
                onCreateTime();
              }}
            />
          )}
          {selection.type === "event" && (
            <EventDetail
              state={state}
              item={selection.item}
              linked={linked}
              sourceLabel={sourceName}
              onInbox={() => onInbox(selection.item.agent_name)}
            />
          )}
          {selection.type === "agent" &&
            (taskAgent ? (
              <TaskPanel
                key={taskAgent.agent_name}
                state={state}
                agent={taskAgent}
                view={selection.view}
                onView={(view) => setSelection({ ...selection, view })}
                draft={drafts[taskAgent.agent_name] ?? ""}
                onDraft={(draft) =>
                  setDrafts((value) => ({
                    ...value,
                    [taskAgent.agent_name]: draft,
                  }))
                }
              >
                <AgentDetail
                  agent={taskAgent}
                  state={state}
                  status={phase(taskAgent, t)}
                  linked={linked}
                  sourceLabel={sourceName}
                  onOutbox={() => openTask(taskAgent.agent_name, "outbox")}
                />
              </TaskPanel>
            ) : (
              <p className="document-loading">{t("flowConnecting")}</p>
            ))}
          {selection.type === "reminder" && (
            <ReminderDetail
              state={state}
              item={selection.item}
              linked={linked}
              sourceLabel={sourceName}
            />
          )}
        </FlowDrawer>
      )}
      {active && menu && (
        <ContextMenu
          position={menu.position}
          items={menu.items}
          onClose={() => setMenu(null)}
        />
      )}
    </div>
  );
}

function DirectoryDetail({
  path,
  state,
  linked,
}: {
  path: string;
  state: StudioState;
  linked: (paths: string[]) => ReactNode;
}) {
  const request = useWorkspace();
  const t = translator(state.settings.language);
  const [files, setFiles] = useState<string[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [limit, setLimit] = useState(25);
  useEffect(() => {
    let live = true;
    void request({ type: "list", path })
      .then((value) => {
        if (live)
          setFiles((value.metadata.items as string[] | undefined) ?? []);
      })
      .catch((reason) => {
        if (live) setError(reason);
      });
    return () => {
      live = false;
    };
  }, [path]);
  return (
    <>
      {error != null ? (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      ) : files === null ? (
        <p className="flow-hint">{t("documentLoading")}</p>
      ) : files.length ? (
        linked(files.slice(0, limit))
      ) : (
        <p className="flow-hint">{t("flowNoFiles")}</p>
      )}
      {files && files.length > limit && (
        <button
          className="secondary"
          onClick={() => setLimit((value) => value + 25)}
        >
          {t("flowViewAll")}
        </button>
      )}
    </>
  );
}
