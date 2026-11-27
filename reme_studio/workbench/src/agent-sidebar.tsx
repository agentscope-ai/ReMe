import { useEffect, useId, useRef, useState } from "react";
import {
  Bell,
  BookOpenText,
  Ellipsis,
  Inbox,
  MessageCircle,
} from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { translator } from "../../client/src/i18n.ts";
import { SidebarHeading } from "./split-pane.tsx";
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
export type RemySection = (typeof remySections)[number][0];

function AgentStatus({
  agent,
  t,
}: {
  agent: AgentView;
  t: ReturnType<typeof translator>;
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
  const label = t(key);
  const error = tone === "error" ? agent.error : null;
  return (
    <span
      className="agent-status"
      data-tone={tone}
      aria-describedby={error ? descriptionId : undefined}
      tabIndex={error ? 0 : undefined}
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
      {label}
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
  section,
  actions,
  onSelect,
}: {
  state: StudioState;
  section: RemySection;
  actions: Record<RemySection, MenuItem[]>;
  onSelect: (section: RemySection) => void;
}) {
  const t = translator(state.settings.language);
  const remy = state.agents?.find((agent) => agent.agent_name === "Remy");
  const active = state.activePage === "remy";
  const [menu, setMenu] = useState<{
    id: RemySection;
    position?: MenuPosition;
  } | null>(null);
  useEffect(() => {
    if (!active) setMenu(null);
  }, [active]);
  const closeMenu = () => setMenu(null);
  return (
    <nav className="agent-sidebar" data-page="remy" aria-label="Remy">
      <SidebarHeading eyebrow="AT YOUR SIDE" title="Remy">
        {remy && <AgentStatus agent={remy} t={t} />}
      </SidebarHeading>
      <div className="agent-navigation">
        <ul
          className="agent-sections"
          aria-label={t("agentPages", { name: "Remy" })}
        >
          {remySections.map(([id, label, Icon]) => {
            const expanded = active && menu?.id === id && !menu.position;
            return (
              <li key={id} className="agent-row-wrapper">
                <div
                  className="agent-row-controls"
                  onContextMenu={(event) => {
                    event.preventDefault();
                    event.currentTarget
                      .querySelector("button")
                      ?.focus({ preventScroll: true });
                    setMenu({
                      id,
                      position: { x: event.clientX, y: event.clientY },
                    });
                  }}
                >
                  <button
                    type="button"
                    className="agent-row"
                    aria-current={section === id ? "page" : undefined}
                    onClick={() => onSelect(id)}
                  >
                    <Icon size={16} aria-hidden />
                    {t(label)}
                    {id === "reminders" && state.unread > 0 && (
                      <span className="count-badge" data-tone="unread">
                        {state.unread}
                      </span>
                    )}
                  </button>
                  <button
                    type="button"
                    className="agent-row-menu"
                    aria-label={t("sectionActions", { name: t(label) })}
                    aria-haspopup="menu"
                    aria-expanded={expanded}
                    onClick={(event) => {
                      event.currentTarget.focus({ preventScroll: true });
                      setMenu((value) =>
                        value?.id === id && !value.position ? null : { id },
                      );
                    }}
                  >
                    <Ellipsis size={16} aria-hidden />
                  </button>
                </div>
                <ContextMenu
                  open={expanded}
                  items={actions[id]}
                  onClose={closeMenu}
                />
              </li>
            );
          })}
        </ul>
      </div>
      {active && menu?.position && (
        <ContextMenu
          key={menu.id}
          position={menu.position}
          items={actions[menu.id]}
          onClose={closeMenu}
        />
      )}
    </nav>
  );
}
