import { messageKey } from "../../../client/src/message-key.ts";
import type {
  StudioState,
  DisplayMessage,
} from "../../../client/src/protocol.ts";
import type { DesktopPatch } from "../protocol.ts";

export const STREAM_RENDER_MS = 33;

/** IPC clones the transcript; restore identity for unchanged rows and items. */
function share<T>(previous: T, incoming: T): T {
  if (Object.is(previous, incoming)) return previous;
  if (
    !previous ||
    !incoming ||
    typeof previous !== "object" ||
    typeof incoming !== "object"
  )
    return incoming;
  if (Array.isArray(previous) && Array.isArray(incoming)) {
    const values = incoming.map((value, index) =>
      share(previous[index], value),
    );
    return (
      values.length === previous.length &&
      values.every((value, index) => value === previous[index])
        ? previous
        : values
    ) as T;
  }
  if (Array.isArray(previous) || Array.isArray(incoming)) return incoming;
  const before = previous as Record<string, unknown>;
  const after = incoming as Record<string, unknown>;
  const values = Object.fromEntries(
    Object.entries(after).map(([key, value]) => [
      key,
      share(before[key], value),
    ]),
  );
  return (
    Object.keys(before).length === Object.keys(after).length &&
    Object.keys(after).every(
      (key) => Object.hasOwn(before, key) && values[key] === before[key],
    )
      ? previous
      : values
  ) as T;
}

export function mergeState(
  state: StudioState,
  patch: DesktopPatch,
): StudioState {
  if (patch.revision <= state.revision) return state;
  const next = { ...state, ...patch };
  if (
    patch.messages &&
    next.sessionId === state.sessionId &&
    next.selectedAgentName === state.selectedAgentName
  ) {
    const previous = new Map(
      state.messages.map((message) => [messageKey(message), message]),
    );
    const messages = patch.messages.map((message) =>
      share(previous.get(messageKey(message)), message)!,
    );
    next.messages =
      messages.length === state.messages.length &&
      messages.every((message, index) => message === state.messages[index])
        ? state.messages
        : messages;
  }
  return next;
}

function boundary(message: DisplayMessage): string {
  return JSON.stringify([
    messageKey(message),
    message.completed,
    message.streaming,
    message.error,
    message.status,
    message.items?.map((item) => [
      item.id,
      item.type,
      item.status,
      item.type === "text" ? item.phase : undefined,
    ]),
  ]);
}

const cosmeticKeys = new Set([
  "revision",
  "messages",
  "pending",
  "surface",
  "behavior",
]);

function continuous(
  previous: StudioState,
  next: StudioState,
  patch: DesktopPatch,
): boolean {
  return Boolean(
    patch.messages &&
    Object.keys(patch).every((key) => cosmeticKeys.has(key)) &&
    previous.sessionId === next.sessionId &&
    previous.selectedAgentName === next.selectedAgentName &&
    previous.pending?.outcome === next.pending?.outcome &&
    previous.pending?.streaming === next.pending?.streaming &&
    previous.messages.length === next.messages.length &&
    next.messages.some((message) => message.streaming) &&
    next.messages.every(
      (message, index) =>
        boundary(message) === boundary(previous.messages[index]),
    ),
  );
}

/** Coalesce cosmetic updates; tools, terminal outcomes and context changes flush immediately. */
export class DesktopUpdates {
  private timer: ReturnType<typeof setTimeout> | undefined;
  private published: StudioState;
  private closed = false;

  constructor(
    private current: StudioState,
    private readonly publish: (state: StudioState) => void,
  ) {
    this.published = current;
  }

  push(patch: DesktopPatch): void {
    if (this.closed) return;
    const previous = this.current;
    this.current = mergeState(previous, patch);
    if (previous === this.current) return;
    if (!continuous(previous, this.current, patch)) this.flush();
    // Timers still run in hidden Electron windows; animation frames may stop entirely.
    else this.timer ??= setTimeout(() => this.flush(), STREAM_RENDER_MS);
  }

  flush = (): void => {
    clearTimeout(this.timer);
    this.timer = undefined;
    if (!this.closed && this.current !== this.published) {
      this.published = this.current;
      this.publish(this.current);
    }
  };

  close(): void {
    this.closed = true;
    clearTimeout(this.timer);
    this.timer = undefined;
  }
}
