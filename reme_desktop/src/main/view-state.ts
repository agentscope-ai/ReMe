import type {
  DesktopPatch,
  DesktopState,
  DisplayMessage,
} from "../shared/protocol.ts";
import { previewContent } from "../shared/surface.ts";

const noMessages: DisplayMessage[] = [];

/** The companion receives the selected preview, never the full history or local prompt. */
export function viewState(
  state: DesktopState,
  companion: boolean,
): DesktopState {
  return companion
    ? {
        ...state,
        messages: noMessages,
        surface:
          state.surface?.kind === "message"
            ? {
                ...state.surface,
                message: {
                  ...state.surface.message,
                  content: previewContent(state.surface.message.content),
                },
              }
            : state.surface,
        readMessageIds: [],
        pending: null,
        workspace: "",
        focusMessageId: null,
        focusRequest: 0,
      }
    : state;
}

export class StateUpdates {
  private readonly previous = new Map<
    string,
    { value: unknown; encoded: string | undefined }
  >();

  constructor(private readonly companion: boolean) {}

  update(state: DesktopState): DesktopPatch | null {
    const changed: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(
      viewState(state, this.companion),
    )) {
      if (key === "revision") continue;
      const previous = this.previous.get(key);
      // SessionModel replaces message arrays atomically; no need to serialize unchanged history.
      if (key === "messages" && previous?.value === value) continue;
      const encoded = JSON.stringify(value);
      this.previous.set(key, { value, encoded });
      if (!previous || previous.encoded !== encoded) changed[key] = value;
    }
    return Object.keys(changed).length
      ? { ...changed, revision: state.revision }
      : null;
  }
}
