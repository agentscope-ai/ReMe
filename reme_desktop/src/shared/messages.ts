import { translator } from "./i18n.ts";
import type { Language } from "./i18n.ts";
import { finishItems } from "./transcript.ts";
import type {
  AgentActivity,
  DisplayMessage,
  PendingTurn,
  TranscriptMessage,
  StatusNotice,
} from "./protocol.ts";

export const terminal = (phase: AgentActivity["phase"]) =>
  ["completed", "failed", "interrupted"].includes(phase);

export function activityNotice(
  activity: AgentActivity,
  language: Language,
): StatusNotice {
  const phase = activity.phase;
  return {
    kind: "status",
    id: `${activity.session_id}:${activity.turn_id}:${phase}`,
    text: translator(language)(phase),
    tone:
      phase === "completed"
        ? "success"
        : phase === "failed" || phase === "interrupted"
          ? "error"
          : phase === "waiting"
            ? "waiting"
            : "working",
  };
}

/** Some providers store each model response separately within one agent turn. */
function groupTurns(messages: TranscriptMessage[]): TranscriptMessage[] {
  const result: TranscriptMessage[] = [];
  const turns = new Map<string, number>();
  for (const message of messages) {
    const key =
      message.role === "assistant" && message.source === "chat"
        ? message.turn_id
        : null;
    const index = key ? turns.get(key) : undefined;
    if (index === undefined) {
      if (key) turns.set(key, result.length);
      result.push(message);
    } else {
      const previous = result[index];
      result[index] = {
        ...previous,
        content: [previous.content, message.content]
          .filter(Boolean)
          .join("\n\n"),
        items: [...previous.items, ...message.items],
        completed: previous.completed && message.completed,
        outcome: message.outcome ?? previous.outcome,
        error: message.error ?? previous.error,
      };
    }
  }
  return result;
}

/** Project native history and the local stream once, before either window renders it. */
export function displayMessages(
  state: {
    sessionId: string | null;
    messages: TranscriptMessage[];
    pending: PendingTurn | null;
    replaceOnSend?: boolean;
  },
  activity: AgentActivity | undefined,
  language: Language,
): DisplayMessage[] {
  const pending =
    state.pending?.sessionId === state.sessionId ? state.pending : null;
  const messages = groupTurns(
    pending && state.replaceOnSend
      ? state.messages.filter(
          (message) => pending.turnId && message.turn_id === pending.turnId,
        )
      : state.messages,
  );
  const result: DisplayMessage[] = messages.map((message) => {
    const live =
      pending?.turnId &&
      message.role === "assistant" &&
      message.turn_id === pending.turnId
        ? pending
        : null;
    const items = message.items;
    const outcome = message.outcome;
    return {
      id: message.id,
      turnId: message.turn_id ?? undefined,
      createdAt: message.created_at,
      originSessionId:
        message.origin_session_id ?? state.sessionId ?? undefined,
      role: message.role,
      content: items.some((item) => item.type === "text")
        ? items
            .filter((item) => item.type === "text")
            .map((item) => item.text)
            .join("\n\n")
        : message.content,
      items:
        message.completed || (live?.outcome && !live.streaming)
          ? finishItems(
              items,
              live?.error
                ? "interrupted"
                : (message.outcome ?? live?.outcome ?? "completed"),
            )
          : items,
      completed: message.completed,
      source: message.source,
      streaming: !message.completed && !(live?.outcome && !live.streaming),
      status:
        outcome && !["completed", "running"].includes(outcome)
          ? {
              kind: "status",
              id: `${message.id}:${outcome}`,
              text: translator(language)(outcome),
              tone: outcome === "waiting" ? "waiting" : "error",
            }
          : null,
      error: live?.error ?? message.error,
    };
  });
  const matches = (role: "user" | "assistant") =>
    pending?.turnId &&
    messages.some(
      (message) => message.role === role && message.turn_id === pending.turnId,
    );
  if (pending && !matches("user"))
    result.push({
      id: `local-user:${pending.id}`,
      role: "user",
      source: "chat",
      turnId: pending.turnId,
      content: pending.query,
      completed: false,
      streaming: false,
      status: null,
    });
  if (pending && !matches("assistant"))
    result.push({
      id: `local-assistant:${pending.id}`,
      role: "assistant",
      source: "chat",
      turnId: pending.turnId,
      content: "",
      items: [],
      completed: false,
      streaming:
        pending.streaming || (!pending.outcome && Boolean(pending.turnId)),
      status: null,
      error: pending.error,
    });
  let current = activity
    ? result.findLast(
        (message) =>
          message.role === "assistant" &&
          message.source === "chat" &&
          message.turnId === activity.turn_id,
      )
    : undefined;
  if (
    activity &&
    !current &&
    pending &&
    !pending.turnId &&
    !terminal(activity.phase)
  )
    current = result.findLast((message) => message.role === "assistant");
  if (
    activity &&
    !current &&
    (!pending || pending.turnId === activity.turn_id)
  ) {
    current = {
      id: `turn:${activity.turn_id}`,
      turnId: activity.turn_id,
      role: "assistant",
      source: "chat",
      content: "",
      completed: false,
      streaming: true,
      status: null,
    };
    result.push(current);
  }
  if (activity && current) {
    current.status = {
      ...activityNotice(activity, language),
      messageId: current.id,
    };
    if (terminal(activity.phase)) current.streaming = false;
  }
  if (pending) {
    const reply = result.findLast(
      (message) =>
        message.role === "assistant" &&
        (pending.turnId
          ? message.turnId === pending.turnId
          : message.id === `local-assistant:${pending.id}`),
    );
    if (reply && !reply.status)
      reply.status = {
        ...activityNotice(
          {
            session_id: state.sessionId!,
            turn_id: pending.turnId ?? pending.id,
            source: "chat",
            phase: pending.streaming
              ? reply.content
                ? "answering"
                : "thinking"
              : pending.outcome === "waiting"
                ? "waiting"
                : pending.outcome === "failed"
                  ? "failed"
                  : pending.outcome === "interrupted"
                    ? "interrupted"
                    : "completed",
          },
          language,
        ),
        messageId: reply.id,
      };
    if (reply?.error)
      reply.status = {
        kind: "status",
        id: `error:${reply.id}`,
        text: reply.error,
        tone: pending.outcome ? "error" : "waiting",
        messageId: reply.id,
      };
  }
  return result;
}
