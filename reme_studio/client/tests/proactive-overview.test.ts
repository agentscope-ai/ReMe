import assert from "node:assert/strict";
import test from "node:test";
import {
  ProactiveOverviewModel,
  OverviewVersionError,
  watchProactiveOverview,
} from "../src/proactive-overview.ts";
import { ReMeClient } from "../src/reme-client.ts";
import type {
  OverviewSnapshot,
  OverviewTransition,
  OverviewUpdate,
} from "../src/wire.ts";

function snapshot(revision = 0, epoch = "one"): OverviewSnapshot {
  const inbox = {
    waiting_count: 0,
    processing_count: 0,
    processed_count: 0,
    queues: {},
    items: [],
  };
  return {
    type: "snapshot",
    epoch,
    revision,
    as_of: "2026-11-27T10:00:00Z",
    overview: {
      sources: [],
      agents: [],
      event_inbox: structuredClone(inbox),
      remy_inbox: structuredClone(inbox),
      reminders: [],
      workspace: [],
      errors: {},
    },
  };
}

function action(id = "admission"): OverviewTransition {
  return {
    id,
    type: "event_admitted",
    occurred_at: "2026-11-27T10:00:01Z",
    source_id: "screen",
    agent_name: "memory",
    target_agent: null,
    event_ids: ["event"],
    run_id: null,
    run_ids: [],
    delivery_id: null,
    message_id: null,
    paths: [],
  };
}

function update(previousRevision = 0, revision = 1): OverviewUpdate {
  return {
    type: "update",
    epoch: "one",
    previous_revision: previousRevision,
    revision,
    as_of: "2026-11-27T10:00:01Z",
    changes: [],
    transitions: [action()],
  };
}

test("overview snapshots, entity replacements and removals apply atomically", () => {
  const model = new ProactiveOverviewModel();
  assert.deepEqual(model.apply(snapshot()), []);
  const next = update();
  next.changes = [
    { collection: "errors", id: "source:screen", value: "offline" },
    {
      collection: "workspace",
      id: "record",
      value: {
        id: "record",
        agent_name: "memory",
        role: "record",
        path: "daily/result.md",
        mtime: null,
        run_id: "run",
      },
    },
  ];
  assert.deepEqual(model.apply(next), [action()]);
  assert.equal(model.overview?.errors["source:screen"], "offline");
  assert.equal(model.overview?.workspace[0].path, "daily/result.md");
  const remove = update(1, 2);
  remove.changes = [
    { collection: "errors", id: "source:screen", value: null },
    { collection: "workspace", id: "record", value: null },
  ];
  assert.deepEqual(model.apply(remove), []);
  assert.deepEqual(model.overview?.workspace, []);
  assert.deepEqual(model.overview?.errors, {});
});

test("newly sent reminders remain first after entity updates", () => {
  const model = new ProactiveOverviewModel();
  const initial = snapshot();
  const reminder = {
    id: "older",
    session_id: "session",
    summary: "Older reminder",
    delivered_at: "2026-11-27T09:00:00Z",
    status: "sent" as const,
    notification_ids: [],
    source_refs: [],
    event_ids: [],
    run_ids: [],
    record_paths: [],
  };
  initial.overview.reminders = [reminder];
  model.apply(initial);
  const next = update();
  next.changes = [
    {
      collection: "reminders",
      id: "newer",
      value: {
        ...reminder,
        id: "newer",
        delivered_at: "2026-11-27T10:00:00Z",
      },
    },
  ];
  model.apply(next);
  assert.deepEqual(
    model.overview?.reminders.map((item) => item.id),
    ["newer", "older"],
  );
});

test("duplicate actions and redelivered versions never animate twice", () => {
  const model = new ProactiveOverviewModel();
  model.apply(snapshot());
  assert.equal(model.apply(update()).length, 1);
  assert.deepEqual(model.apply(update()), []);
  assert.deepEqual(model.apply(update(1, 2)), []);
  model.live = false;
  assert.deepEqual(model.apply(snapshot(2)), []);
  assert.deepEqual(model.apply(update(2, 3)), []);
  assert.equal(model.live, true);
});

test("version gaps, epoch changes and malformed changes preserve the last valid state", () => {
  const model = new ProactiveOverviewModel();
  model.apply(snapshot());
  assert.throws(() => model.apply(update(2, 3)), OverviewVersionError);
  assert.throws(
    () => model.apply({ ...update(), epoch: "restarted" }),
    OverviewVersionError,
  );
  const invalid = update();
  invalid.changes = [
    { collection: "errors", id: "first", value: "should not commit" },
    {
      collection: "workspace",
      id: "wrong",
      value: {
        id: "other",
        agent_name: null,
        role: "record",
        path: "daily/result.md",
        mtime: null,
        run_id: null,
      },
    },
  ];
  assert.throws(() => model.apply(invalid));
  assert.equal(model.revision, 0);
  assert.deepEqual(model.overview?.errors, {});
  assert.deepEqual(model.apply(snapshot(0, "restarted")), []);
  assert.equal(model.epoch, "restarted");
});

test("reconnect retains the last snapshot and suppresses historical animations", async () => {
  const controller = new AbortController();
  let requests = 0;
  const states: { live: boolean; revision: number; actions: number }[] = [];
  const client = new ReMeClient(
    "http://127.0.0.1:2333",
    async (_input, init) => {
      requests += 1;
      assert.deepEqual(JSON.parse(init?.body as string), {
        agent_names: ["memory"],
      });
      const events =
        requests === 1 ? [snapshot(), update()] : [snapshot(4), update(4, 5)];
      const body = events
        .map(
          (event) =>
            `data: ${JSON.stringify({ chunk_type: "data", chunk: event })}\n\n`,
        )
        .join("");
      return new Response(body, {
        headers: { "Content-Type": "text/event-stream" },
      });
    },
  );
  await watchProactiveOverview(client, {
    signal: controller.signal,
    agentNames: ["memory"],
    onChange(model, actions) {
      states.push({
        live: model.live,
        revision: model.revision,
        actions: actions.length,
      });
      if (model.revision === 5) controller.abort();
    },
  });
  assert.equal(requests, 2);
  assert.ok(states.some((state) => !state.live && state.revision === 1));
  assert.equal(
    states.reduce((total, state) => total + state.actions, 0),
    1,
  );
  assert.equal(states.at(-1)?.live, false);
});

test("reduced motion drops animations while applying every state change", async () => {
  const controller = new AbortController();
  const client = new ReMeClient(
    "http://127.0.0.1:2333",
    async () =>
      new Response(
        [snapshot(), update()]
          .map(
            (event) =>
              `data: ${JSON.stringify({ chunk_type: "data", chunk: event })}\n\n`,
          )
          .join(""),
        { headers: { "Content-Type": "text/event-stream" } },
      ),
  );
  await watchProactiveOverview(client, {
    signal: controller.signal,
    animations: () => false,
    onChange(model, actions) {
      assert.deepEqual(actions, []);
      if (model.revision === 1) controller.abort();
    },
  });
});
