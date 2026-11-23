import { useEffect, useState } from "react";
import { CheckCheck } from "lucide-react";
import type { StudioState } from "../../client/src/protocol.ts";
import { errorText, translator } from "../../client/src/i18n.ts";
import { useHost } from "./host.ts";
import { plainText } from "./resources.ts";
import ResourcePane, { ResourceMessage } from "./resource-pane.tsx";
import { watchReadReceipt } from "./read-receipt.ts";

export default function Reminders({ state }: { state: StudioState }) {
  const { action } = useHost();
  const t = translator(state.settings.language);
  const [selected, setSelected] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [marking, setMarking] = useState(false);
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
        rows={state.reminders.toReversed().map((item) => ({
          id: item.id,
          title: "Remy",
          preview: plainText(item.content),
          time: item.createdAt,
          unread: !state.readMessageIds.includes(item.id),
          badge: t(
            state.readMessageIds.includes(item.id)
              ? "inboxUserRead"
              : "inboxUserUnread",
          ),
          target: { kind: "message", id: item.id },
        }))}
        controls={
          <button
            className="subtle"
            disabled={
              marking || state.connection !== "connected" || !state.unread
            }
            onClick={() => {
              setMarking(true);
              void action({ type: "items.read" })
                .catch(setError)
                .finally(() => setMarking(false));
            }}
          >
            <CheckCheck size={15} />
            {t("markAllRead")}
          </button>
        }
      >
        {message && (
          <ResourceMessage
            state={state}
            message={message}
            target={{ kind: "message", id: message.id }}
          />
        )}
      </ResourcePane>
    </>
  );
}
