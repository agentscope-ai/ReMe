import assert from "node:assert/strict";
import test from "node:test";
import { viewState } from "../src/main/view-state.ts";
import type { DesktopState, DisplayMessage } from "../src/shared/protocol.ts";

test("the companion receives final text without tool details or full history", () => {
  const message: DisplayMessage = {
    id: "reply",
    role: "assistant",
    source: "chat",
    content: "All progress and final text",
    streaming: false,
    completed: true,
    status: null,
    items: [
      {
        id: "intro",
        type: "text",
        text: "Checking",
        status: "completed",
        phase: "commentary",
      },
      {
        id: "tool",
        type: "tool",
        name: "bash",
        input: "private arguments",
        output: "large tool output",
        status: "completed",
        actions: [{ kind: "command" }],
      },
      {
        id: "answer",
        type: "text",
        text: "Done",
        status: "completed",
        phase: "final_answer",
      },
    ],
  };
  const state = {
    agents: [],
    sessionId: "session",
    messages: [message],
    surface: {
      kind: "message",
      message,
      presentationId: 1,
      target: { kind: "message", id: message.id },
    },
  } as DesktopState;
  const companion = viewState(state, true);
  assert.deepEqual(companion.messages, []);
  assert.equal(companion.surface?.kind, "message");
  if (companion.surface?.kind === "message") {
    assert.equal(companion.surface.message.content, "Done");
    assert.equal(companion.surface.message.items, undefined);
  }
  assert.equal(JSON.stringify(companion).includes("private arguments"), false);
  assert.equal(viewState(state, false), state);
});
