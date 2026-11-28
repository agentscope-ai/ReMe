import { useId, useRef } from "react";
import { Bell, BookOpenText, Inbox, MessageCircle } from "lucide-react";
import type { AgentView, StudioState } from "../../client/src/protocol.ts";
import { translator } from "../../client/src/i18n.ts";
import { SidebarHeading } from "./split-pane.tsx";

export const remySections = [
  ["chat", "conversationTab", MessageCircle],
  ["reminders", "reminders", Bell],
  ["document", "memoryTab", BookOpenText],
  ["inbox", "inbox", Inbox],
] as const;
export type RemySection = (typeof remySections)[number][0];

export function AgentStatus({
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
  const label = t(key);
  const error = tone === "error" ? agent.error : null;
  return (
    <span
      className="agent-status"
      data-tone={tone}
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
      {!compact && label}
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
  onSelect,
}: {
  state: StudioState;
  section: RemySection;
  onSelect: (section: RemySection) => void;
}) {
  const t = translator(state.settings.language);
  const remy = state.agents?.find((agent) => agent.agent_name === "Remy");
  return (
    <nav className="agent-sidebar" data-page="remy" aria-label="Remy">
      <SidebarHeading title="Remy">
        {remy && <AgentStatus agent={remy} t={t} />}
      </SidebarHeading>
      <div className="agent-navigation">
        <ul
          className="agent-sections"
          aria-label={t("agentPages", { name: "Remy" })}
        >
          {remySections.map(([id, label, Icon]) => (
            <li key={id} className="agent-row-wrapper">
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
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
