import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import {
  Bell,
  BookOpenText,
  Ellipsis,
  FileText,
  FolderOpen,
  Inbox,
  MessageCircle,
  Pause,
  Play,
  Plus,
  Send,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { translator } from "../../client/src/i18n.ts";
import ContextMenu, {
  type MenuItem,
  type MenuPosition,
} from "./context-menu.tsx";

export const remySections = [
  ["chat", "conversationTab", MessageCircle],
  ["reminders", "reminders", Bell],
  ["document", "memoryTab", BookOpenText],
  ["inbox", "inbox", Inbox],
] as const;
export const taskSections = [
  ["manage", "taskManage", SlidersHorizontal],
  ["document", "taskFileTab", FileText],
  ["inbox", "inbox", Inbox],
  ["outbox", "outbox", Send],
] as const;
export type RemySection = (typeof remySections)[number][0];
export type AgentSection =
  RemySection | "create" | (typeof taskSections)[number][0];

interface SidebarRow {
  id: string;
  content: ReactNode;
  title?: string;
  current?: true | "page";
  disabled?: boolean;
  select: () => void;
  menuLabel: string;
  items: MenuItem[];
}

function AgentStatus({
  agent,
  t,
  compact = false,
}: {
  agent: AgentView;
  t: ReturnType<typeof translator>;
  compact?: boolean;
}) {
  const [tone, key] = !agent.enabled
    ? (["disabled", "agentDisabled"] as const)
    : agent.error || agent.status === "error"
      ? (["error", "agentError"] as const)
      : agent.status === "active"
        ? (["running", "agentRunning"] as const)
        : (["idle", "agentIdle"] as const);
  const descriptionId = useId();
  const detail = useRef<
    HTMLSpanElement & { showPopover(options: { source: HTMLElement }): void }
  >(null);
  if (compact && tone === "idle") return null;
  const label = t(key);
  const error = tone === "error" ? agent.error : null;
  return (
    <span
      className="agent-status"
      data-tone={tone}
      data-badge={!compact || undefined}
      role={compact ? "img" : undefined}
      aria-label={compact ? label : undefined}
      aria-describedby={error ? descriptionId : undefined}
      tabIndex={error && !compact ? 0 : undefined}
      title={
        error
          ? undefined
          : !agent.enabled && agent.status === "active"
            ? t("taskPauseHint")
            : agent.error || label
      }
      onPointerEnter={
        error
          ? (event) =>
              detail.current?.showPopover({ source: event.currentTarget })
          : undefined
      }
      onPointerLeave={
        error
          ? (event) => {
              if (!event.currentTarget.matches(":focus-visible"))
                detail.current?.hidePopover();
            }
          : undefined
      }
      onFocus={
        error
          ? (event) =>
              detail.current?.showPopover({ source: event.currentTarget })
          : undefined
      }
      onBlur={error ? () => detail.current?.hidePopover() : undefined}
    >
      {compact ? <i aria-hidden /> : label}
      {error && (
        <span
          ref={detail}
          id={descriptionId}
          className="agent-status-detail"
          popover="hint"
          role="tooltip"
          onClick={(event) => event.stopPropagation()}
        >
          {error}
        </span>
      )}
    </span>
  );
}

export default function AgentSidebar({
  state,
  page,
  section,
  remyActions,
  onSelect,
  onDelete,
  onToggle,
  onReveal,
  busy,
}: {
  state: StudioState;
  page: "remy" | "tasks";
  section: AgentSection;
  remyActions: Record<RemySection, MenuItem[]>;
  onSelect: (agentName: string | null, section: AgentSection) => void;
  onDelete: (agentName: string) => void;
  onToggle: (agent: AgentView) => void;
  onReveal: (agentName: string) => void;
  busy: boolean;
}) {
  const t = translator(state.settings.language);
  const selected = state.selectedAgentName ?? "Remy";
  const agents = state.agents ?? [];
  const remy = agents.find((agent) => agent.agent_name === "Remy");
  const tasks = agents.filter((agent) => agent.agent_type !== "chat");
  const active = state.activePage === page;
  const [menu, setMenu] = useState<{
    rowId: string;
    position?: MenuPosition;
  } | null>(null);
  const closeMenu = () => setMenu(null);
  useEffect(() => {
    if (!active) setMenu(null);
  }, [active]);
  const taskMenuItems = (agent: AgentView): MenuItem[] => [
    ...taskSections.map(([section, label, icon]) => ({
      id: section,
      label: t(label),
      icon,
      select: () => onSelect(agent.agent_name, section),
    })),
    {
      id: "toggle",
      label: t(agent.enabled ? "taskPause" : "taskEnable"),
      icon: agent.enabled ? Pause : Play,
      disabled: busy || state.connection !== "connected",
      select: () => onToggle(agent),
    },
    {
      id: "directory",
      label: t("taskOpenDirectory"),
      icon: FolderOpen,
      select: () => onReveal(agent.agent_name),
    },
    {
      id: "delete",
      label: t("taskDelete"),
      icon: Trash2,
      danger: true,
      disabled: busy,
      select: () => onDelete(agent.agent_name),
    },
  ];
  const rows: SidebarRow[] =
    page === "remy"
      ? remySections.map(([id, label, Icon]) => ({
          id,
          current: section === id ? "page" : undefined,
          select: () => onSelect("Remy", id),
          menuLabel: t("sectionActions", { name: t(label) }),
          items: remyActions[id],
          content: (
            <>
              <Icon size={16} aria-hidden />
              {t(label)}
              {id === "reminders" && state.unread > 0 && (
                <span className="count-badge" data-tone="accent">
                  {state.unread}
                </span>
              )}
            </>
          ),
        }))
      : tasks.map((agent) => {
          const name = agent.agent_name;
          const timeTriggered = agent.agent_type === "time";
          return {
            id: name,
            title: name,
            current: selected === name || undefined,
            disabled: !agent.enabled,
            select: () => onSelect(name, "chat"),
            menuLabel: t("taskMenu", { name }),
            items: taskMenuItems(agent),
            content: (
              <>
                <span
                  className="agent-trigger"
                  data-trigger={agent.agent_type}
                  title={t(
                    timeTriggered ? "taskTimeTrigger" : "taskEventTrigger",
                  )}
                >
                  {timeTriggered ? "Time" : "Event"}
                </span>
                <span className="agent-name">{name}</span>
                <AgentStatus agent={agent} t={t} compact />
              </>
            ),
          };
        });
  const menuRow = rows.find(({ id }) => id === menu?.rowId);
  const row = (entry: SidebarRow) => {
    const expanded = active && menu?.rowId === entry.id && !menu.position;
    return (
      <li
        key={entry.id}
        className="agent-row-wrapper"
        data-disabled={entry.disabled || undefined}
      >
        <div
          className="agent-row-controls"
          onContextMenu={(event) => {
            event.preventDefault();
            event.currentTarget
              .querySelector("button")
              ?.focus({ preventScroll: true });
            const bounds = event.currentTarget.getBoundingClientRect();
            setMenu({
              rowId: entry.id,
              position: {
                x: event.clientX || bounds.left,
                y: event.clientY || bounds.bottom,
              },
            });
          }}
        >
          <button
            type="button"
            className="agent-row"
            aria-current={entry.current}
            title={entry.title}
            onClick={entry.select}
          >
            {entry.content}
          </button>
          <button
            type="button"
            className="agent-row-menu"
            aria-label={entry.menuLabel}
            aria-haspopup="menu"
            aria-expanded={expanded}
            onClick={(event) => {
              event.currentTarget.focus({ preventScroll: true });
              setMenu((current) =>
                current?.rowId === entry.id && !current.position
                  ? null
                  : { rowId: entry.id },
              );
            }}
          >
            <Ellipsis size={16} aria-hidden />
          </button>
        </div>
        <ContextMenu open={expanded} items={entry.items} onClose={closeMenu} />
      </li>
    );
  };
  return (
    <nav
      className="agent-sidebar"
      data-page={page}
      aria-label={page === "remy" ? "Remy" : t("proactiveAi")}
    >
      <div className="agent-navigation">
        <div className="agent-sidebar-heading">
          <div className="agent-identity">
            <h2>{page === "remy" ? "Remy" : t("proactiveAi")}</h2>
            {page === "remy" ? (
              remy && <AgentStatus agent={remy} t={t} />
            ) : (
              <button
                type="button"
                className="icon-button agent-task-add"
                aria-label={t("taskCreate")}
                title={t("taskCreate")}
                disabled={busy || state.connection !== "connected"}
                onClick={() => onSelect(null, "create")}
              >
                <Plus size={16} aria-hidden />
              </button>
            )}
          </div>
          {page === "remy" && <p>{t("personalAssistant")}</p>}
        </div>
        {rows.length ? (
          <ul
            className={page === "remy" ? "agent-sections" : "agent-list"}
            aria-label={
              page === "remy" ? t("agentPages", { name: "Remy" }) : undefined
            }
          >
            {rows.map(row)}
          </ul>
        ) : (
          <p className="agent-list-empty">{t("noAgents")}</p>
        )}
      </div>
      {active && menu?.position && menuRow && (
        <ContextMenu
          key={menuRow.id}
          position={menu.position}
          items={menuRow.items}
          onClose={closeMenu}
        />
      )}
    </nav>
  );
}
