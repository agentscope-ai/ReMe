import type {
  AgentMessage,
  AgentView,
  StreamChunk,
} from "../../client/src/protocol.ts";

import type { AgentSnapshot } from "../../client/src/wire.ts";

const main = (sessionId: string): AgentView => ({
  agent_name: "Remy",
  agent_type: "chat",
  directory: "reme_agents/Remy",
  memory_path: "reme_agents/Remy/MEMORY.md",
  task_path: null,
  session_id: sessionId,
  status: "idle",
  enabled: true,
  cron: null,
  min_trigger_interval_seconds: 0,
  notify_window: null,
  pending_events: 0,
  in_flight_events: 0,
  last_started_at: null,
  last_completed_at: null,
  next_trigger_at: null,
  error: null,
});

const files: Record<string, string> = {
  "daily/weekly-review.md":
    "# Weekly review\n\n- Summarize completed work\n- List the next steps.",
  "knowledge/reading-list.md":
    "# Reading list\n\nCompare datasets for long-term memory evaluation.",
  "reme_agents/Remy/MEMORY.md":
    "# Remy\n\nStart with a short conclusion and include the next step.",
};
const reply =
  "This is a sample reply. Try opening the reading list in Knowledge.";
const ok = (answer: unknown, metadata = {}) =>
  Response.json({ success: true, answer, metadata });
const encoder = new TextEncoder();
const frame = (chunk: StreamChunk | "[DONE]") =>
  encoder.encode(
    "data: " +
      (typeof chunk === "string" ? chunk : JSON.stringify(chunk)) +
      "\n\n",
  );
const sse = (body: ReadableStream<Uint8Array>) =>
  new Response(body, { headers: { "Content-Type": "text/event-stream" } });

export function demoService(): typeof fetch {
  let sessionId = "demo-1";
  let sequence = 1;
  let messages: AgentMessage[] = [];
  const subscribers = new Set<ReadableStreamDefaultController<Uint8Array>>();
  const snapshot = (): AgentSnapshot => ({
    type: "snapshot",
    agent: main(sessionId),
    agents: [main(sessionId)],
    activities: {},
    messages,
  });
  const publish = () => {
    for (const stream of subscribers)
      stream.enqueue(frame({ chunk_type: "data", chunk: snapshot() }));
  };
  return async (input, init) => {
    const endpoint = String(input).split("/").pop();
    const payload = JSON.parse(String(init?.body ?? "{}"));
    switch (endpoint) {
      case "get_client_info":
        return ok({
          workspace_dir: "/demo",
          timezone: null,
          resource_dir: "resources",
          daily_dir: "daily",
          digest_dir: "knowledge",
          tag_key: "memory_tags",
          jobs: [
            "watch_agent_session",
            "ask_agent",
            "create_chat_session",
            "list_task_reminders",
            "read_agent_reminder",
            "list_agent_inbox",
            "list_workspace",
            "load_file",
            "search_workspace",
            "todo",
            "idea",
          ],
          asr: false,
          tts: false,
          speech_revision: "",
        });
      case "watch_agent_session": {
        let stream: ReadableStreamDefaultController<Uint8Array>;
        let heartbeat: ReturnType<typeof setInterval>;
        const close = () => {
          subscribers.delete(stream);
          clearInterval(heartbeat);
        };
        return sse(
          new ReadableStream({
            start(controller) {
              stream = controller;
              subscribers.add(controller);
              controller.enqueue(
                frame({ chunk_type: "data", chunk: snapshot() }),
              );
              heartbeat = setInterval(
                () =>
                  controller.enqueue(
                    frame({ chunk_type: "data", chunk: { type: "heartbeat" } }),
                  ),
                10_000,
              );
              init?.signal?.addEventListener(
                "abort",
                () => {
                  if (subscribers.has(controller)) {
                    close();
                    controller.close();
                  }
                },
                { once: true },
              );
            },
            cancel: close,
          }),
        );
      }
      case "create_chat_session":
        sessionId = "demo-" + ++sequence;
        messages = [];
        publish();
        return ok(snapshot());
      case "ask_agent": {
        const id = "demo-turn-" + ++sequence;
        const base = {
          session_id: sessionId,
          turn_id: id,
          source: "chat",
          completed: true,
          delivery: "idle",
          origin_session_id: null,
          notification_eligible: false,
          outcome: "completed",
          error: null,
          created_at: new Date().toISOString(),
        } as const;
        const assistant: AgentMessage = {
          ...base,
          id: id + ":assistant",
          role: "assistant",
          content: reply,
          items: [
            {
              id: "answer",
              type: "text",
              text: reply,
              status: "completed",
              phase: "final_answer",
            },
          ],
        };
        messages = [
          ...messages,
          {
            ...base,
            id: id + ":user",
            role: "user",
            content: String(payload.query),
            items: [],
          },
          assistant,
        ];
        publish();
        return sse(
          new ReadableStream({
            start(stream) {
              stream.enqueue(frame({ chunk_type: "data", chunk: assistant }));
              stream.enqueue(frame("[DONE]"));
              stream.close();
            },
          }),
        );
      }
      case "list_workspace":
        return ok("", {
          items: Object.keys(files).filter(
            (path) => !payload.path || path.startsWith(payload.path + "/"),
          ),
        });
      case "load_file":
        return payload.path in files
          ? ok(files[payload.path])
          : Response.json({ success: false, answer: "File not found." });
      case "search_workspace":
        return ok("", {
          results: Object.keys(files)
            .filter((path) =>
              (path + files[path])
                .toLowerCase()
                .includes(String(payload.query).toLowerCase()),
            )
            .map((path) => ({ path })),
        });
      case "todo":
      case "idea":
        return payload.action === "list"
          ? ok({ items: [], total: 0 })
          : Response.json({
              success: false,
              answer: "The demo workspace is read-only.",
            });
      case "list_task_reminders":
      case "list_agent_inbox":
        return ok({ items: [], total: 0 });
      default:
        return Response.json({
          success: false,
          answer: "This job is unavailable in the demo.",
        });
    }
  };
}
