import { useEffect, useState } from "react";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import ResourcePane, {
  ResourceMessage,
  resourceSummary,
} from "./resource-pane.tsx";
import { TimelineBadge } from "./timeline.tsx";
import { watchReadReceipt } from "./read-receipt.ts";
import { useNavigationState } from "./navigation.tsx";

export default function Reminders({ state }: { state: StudioState }) {
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [selected, setSelected] = useNavigationState("remy", "reminder", "");
  const [error, setError] = useState<unknown>(null);
  const message = state.reminders.find((item) => item.id === selected);
  const read = state.readMessageIds.includes(selected);
  useEffect(() => {
    if (state.focusTarget?.kind === "message")
      setSelected(state.focusTarget.id);
  }, [state.focusTarget, state.focusRequest]);
  useEffect(() => {
    if (
      !message?.originSessionId ||
      read ||
      !state.chatVisible ||
      !state.chatFocused ||
      state.activePage !== "remy"
    )
      return;
    const sessionId = message.originSessionId;
    const receipt = watchReadReceipt(async () => {
      await action({ type: "messages.read", sessionId, ids: [message.id] });
      return true;
    }, setError);
    return receipt.close;
  }, [
    message?.id,
    message?.originSessionId,
    read,
    state.chatVisible,
    state.chatFocused,
    state.activePage,
  ]);
  return (
    <>
      {error != null && (
        <p className="action-error" role="alert">
          {errorText(error, state.settings.language)}
        </p>
      )}
      <ResourcePane
        state={state}
        selected={selected}
        onSelect={setSelected}
        view="reminders"
        rows={state.reminders.toReversed().map((item) => {
          const unread = !state.readMessageIds.includes(item.id);
          return {
            id: item.id,
            title: resourceSummary(item.content, "Remy"),
            date: item.createdAt,
            unread,
            metadata: (
              <>
                <span>Remy</span>
                <TimelineBadge tone={unread ? "accent" : undefined}>
                  {t(unread ? "inboxUserUnread" : "inboxUserRead")}
                </TimelineBadge>
              </>
            ),
            target: { kind: "message", id: item.id },
          };
        })}
      >
        {message && (
          <ResourceMessage
            state={state}
            key={message.id}
            message={message}
            sender="Remy"
            metadata={
              <TimelineBadge tone={read ? undefined : "accent"}>
                {t(read ? "inboxUserRead" : "inboxUserUnread")}
              </TimelineBadge>
            }
            target={{ kind: "message", id: message.id }}
          />
        )}
      </ResourcePane>
    </>
  );
}
