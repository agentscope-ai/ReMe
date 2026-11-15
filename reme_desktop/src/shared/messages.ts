import { translator } from "./i18n.ts";
import type { Language } from "./i18n.ts";
import { reduceReply } from "./stream.ts";
import type { ReplyProgress } from "./stream.ts";
import type {
  AgentActivity,
  DisplayMessage,
  PendingTurn,
  SessionMessage,
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

/** Project native history and the local stream once, before either window renders it. */
export function displayMessages(
  state: {
    sessionId: string | null;
    messages: SessionMessage[];
    pending: PendingTurn | null;
    replaceOnSend?: boolean;
  },
  activity: AgentActivity | undefined,
  language: Language,
): DisplayMessage[] {
  const pending =
    state.pending?.sessionId === state.sessionId ? state.pending : null;
  const messages =
    pending && state.replaceOnSend
      ? state.messages.filter(
          (message) => pending.turnId && message.turn_id === pending.turnId,
        )
      : state.messages;
  const result: DisplayMessage[] = messages.map((message) => {
    const live =
      pending?.turnId &&
      message.role === "assistant" &&
      message.turn_id === pending.turnId
        ? pending
        : null;
    const history = message.chunks.reduce<ReplyProgress>(reduceReply, {
      text: "",
      blocks: new Map(),
    });
    const blocks =
      !message.completed && live?.blocks?.length
        ? live.blocks
        : [...history.blocks.values()];
    return {
      id: message.id,
      turnId: message.turn_id ?? undefined,
      createdAt: message.created_at,
      originSessionId:
        message.origin_session_id ?? state.sessionId ?? undefined,
      role: message.role,
      content: blocks.length
        ? blocks.join("\n\n")
        : !message.completed && live?.text
          ? live.text
          : message.content,
      blocks,
      completed: message.completed,
      source: message.source,
      streaming: live
        ? live.streaming && !message.completed
        : !message.completed,
      status: null,
      error: live?.error,
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
      content: pending.text,
      blocks: pending.blocks,
      completed: false,
      streaming: pending.streaming,
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
              ? pending.text
                ? "answering"
                : "thinking"
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
        tone: "error",
        messageId: reply.id,
      };
  }
  return result;
}
