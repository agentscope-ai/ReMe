import assert from "node:assert/strict";
import test from "node:test";
import { ClientController } from "../src/controller.ts";
import { defaultSettings, type ClientStorage } from "../src/settings.ts";
import type { Presentation } from "../src/protocol.ts";
import type { OverviewSnapshot, OverviewUpdate } from "../src/wire.ts";
import { agent, clientInfo, snapshot } from "./fixtures.ts";

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(condition: () => boolean) {
  for (let i = 0; i < 100; i++) {
    if (condition()) return;
    await tick();
  }
  assert.fail("subscription did not reach the expected state");
}

test("overview subscriptions follow page, backend and controller lifecycles without acknowledging events", async () => {
  const presentations = new Map<string, Presentation>();
  const storage: ClientStorage = {
    settings: { ...defaultSettings },
    scope: (endpoint, workspace) => JSON.stringify([endpoint, workspace]),
    presentation: (key) => {
      if (!presentations.has(key))
        presentations.set(key, { initialized: false, receipts: {} });
      return presentations.get(key)!;
    },
    changed() {},
    async flush() {},
  };
  const overview: OverviewSnapshot = {
    type: "snapshot",
    epoch: "one",
    revision: 0,
    as_of: "2026-11-28T08:00:00Z",
    overview: {
      sources: [],
      agents: [],
      reminders: [],
      workspace: [],
      errors: {},
      event_inbox: {
        waiting_count: 0,
        processing_count: 0,
        processed_count: 0,
        queues: {},
        items: [],
      },
      remy_inbox: {
        waiting_count: 0,
        processing_count: 0,
        processed_count: 0,
        queues: {},
        items: [],
      },
    },
  };
  const requests: {
    url: string;
    body: Record<string, unknown>;
    signal: AbortSignal;
  }[] = [];
  const streams: ReadableStreamDefaultController<Uint8Array>[] = [];
  const encode = (event: unknown) =>
    new TextEncoder().encode(
      `data: ${JSON.stringify({ chunk_type: "data", chunk: event })}\n\n`,
    );
  const controller = new ClientController(
    storage,
    () => {},
    undefined,
    undefined,
    undefined,
    async (input, init) => {
      const url = String(input);
      const signal = init!.signal!;
      const body = JSON.parse(init!.body as string);
      requests.push({ url, body, signal });
      if (url.endsWith("/get_client_info"))
        return Response.json({
          success: true,
          answer: {
            ...clientInfo,
            jobs: [...clientInfo.jobs, "watch_proactive_overview"],
          },
          metadata: {},
        });
      const isOverview = url.endsWith("/watch_proactive_overview");
      assert.ok(
        isOverview || url.endsWith("/watch_agent_session"),
        "only read-only jobs are needed",
      );
      const session = snapshot([], "session", String(body.agent_name));
      session.agents = [agent(), agent("monitor")];
      return new Response(
        new ReadableStream<Uint8Array>({
          start(stream) {
            if (isOverview) streams.push(stream);
            stream.enqueue(encode(isOverview ? overview : session));
            signal.addEventListener(
              "abort",
              () => stream.error(signal.reason),
              { once: true },
            );
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } },
      );
    },
  );
  controller.start();
  try {
    await until(() => controller.state.connection === "connected");
    assert.equal(streams.length, 0);
    controller.selectPage("tasks");
    await until(() => controller.state.proactive?.live === true);
    assert.equal(streams.length, 1);
    controller.selectPage("tasks");
    await tick();
    assert.equal(streams.length, 1, "re-renders do not duplicate streams");
    const update: OverviewUpdate = {
      type: "update",
      epoch: "one",
      previous_revision: 0,
      revision: 1,
      as_of: "2026-11-28T08:00:01Z",
      changes: [
        {
          collection: "errors",
          id: "source:screen",
          value: "temporarily offline",
        },
      ],
      transitions: [
        {
          id: "admitted",
          type: "event_admitted",
          occurred_at: "2026-11-28T08:00:01Z",
          source_id: "screen",
          agent_name: "monitor",
          target_agent: null,
          event_ids: ["event"],
          run_id: null,
          run_ids: [],
          delivery_id: null,
          message_id: null,
          paths: [],
        },
      ],
    };
    streams[0].enqueue(encode(update));
    streams[0].enqueue(
      encode({
        ...update,
        previous_revision: 1,
        revision: 2,
        changes: [
          {
            collection: "errors",
            id: "source:screen",
            value: "recovered soon",
          },
        ],
        transitions: [],
      }),
    );
    await until(
      () =>
        controller.state.proactive?.overview?.errors["source:screen"] ===
        "recovered soon",
    );
    assert.equal(
      controller.state.proactive?.actions.length,
      1,
      "fast state-only updates cannot erase an action before the desktop renderer receives it",
    );
    const last = controller.state.proactive!.overview;
    controller.selectPage("knowledge");
    assert.equal(
      requests.find((item) => item.url.endsWith("/watch_proactive_overview"))
        ?.signal.aborted,
      true,
    );
    assert.equal(controller.state.proactive?.live, false);
    assert.equal(controller.state.proactive?.overview, last);
    assert.deepEqual(controller.state.proactive?.actions, []);
    controller.selectPage("tasks");
    await until(() => controller.state.proactive?.live === true);
    assert.equal(streams.length, 2);
    assert.deepEqual(
      controller.state.proactive?.actions,
      [],
      "returning to a page only calibrates its snapshot",
    );
    await controller.settings({ endpoint: "http://127.0.0.1:2444" });
    assert.equal(
      controller.state.proactive,
      undefined,
      "changing backend clears the old workspace projection",
    );
    await until(() =>
      requests.some(
        (item) => item.url === "http://127.0.0.1:2444/watch_proactive_overview",
      ),
    );
  } finally {
    await controller.close();
  }
  assert.ok(
    requests
      .filter((item) => item.url.includes("/watch_"))
      .every((item) => item.signal.aborted),
  );
});
