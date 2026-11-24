import { translator } from "./i18n.ts";
import type { Language } from "./i18n.ts";
import { finishItems } from "./transcript.ts";
import type { ReplyProgress } from "./transcript.ts";
import type {
  ClientActivity,
  DisplayMessage,
  PendingTurn,
  AgentMessage,
  StatusNotice,
} from "./protocol.ts";

export const terminal = (phase: ClientActivity["phase"]) =>
  ["completed", "failed", "interrupted"].includes(phase);

export function activityNotice(
  activity: ClientActivity,
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

/** Project native history and the local stream once, before either window renders it. */
export function displayMessages(
  state: {
    sessionId: string | null;
    messages: AgentMessage[];
    pending: PendingTurn | null;
    reply?: ReplyProgress;
    task?: boolean;
  },
  sessionActivity: ClientActivity | undefined,
  language: Language,
): DisplayMessage[] {
  const activity =
    sessionActivity?.session_id === state.sessionId &&
    sessionActivity.source !== "task"
      ? sessionActivity
      : undefined;
  const pending =
    state.pending?.sessionId === state.sessionId ? state.pending : null;
  const messages = state.messages;
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
      taskInput: state.task && message.role === "user",
      content: message.content,
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
      error: live?.error ?? message.error ?? undefined,
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
      content: state.reply?.content ?? "",
      items: state.reply
        ? pending.streaming
          ? state.reply.items
          : finishItems(state.reply.items, pending.outcome ?? "interrupted")
        : [],
      completed: Boolean(state.reply && !pending.streaming),
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
    activity.phase !== "completed" &&
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
            tool: null,
            started_at: null,
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
