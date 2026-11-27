import assert from "node:assert/strict";
import test from "node:test";
import { connectorPayload, type ConnectorRequest } from "../src/connectors.ts";
import { requestWorkspace } from "../src/workspace.ts";
import type { ReMeClient } from "../src/reme-client.ts";

test("source updates preserve false and empty targets without sending omitted settings", () => {
  assert.deepEqual(
    connectorPayload({
      type: "connector",
      action: "update",
      source_id: "preset:screen_capture_cron",
      enabled: false,
    }),
    {
      action: "update",
      source_id: "preset:screen_capture_cron",
      enabled: false,
    },
  );
  assert.deepEqual(
    connectorPayload({
      type: "connector",
      action: "update",
      source_id: "preset:screen_capture_cron",
      name: " 项目动态 ",
      runtime_names: [],
    }),
    {
      action: "update",
      source_id: "preset:screen_capture_cron",
      name: "项目动态",
      runtime_names: [],
    },
  );
});

test("custom sources use the single SSE creation contract with unicode targets", async () => {
  const calls: [string, unknown][] = [];
  const client = {
    request: async (job: string, body: unknown) => {
      calls.push([job, body]);
      return { success: true, answer: {}, metadata: {} };
    },
  } as unknown as ReMeClient;
  await requestWorkspace(
    client,
    {
      type: "connector",
      action: "create",
      name: "项目动态",
      runtime_names: ["项目跟进", "notes"],
      binding: { url: "https://example.com/events" },
    },
    new AbortController().signal,
  );
  assert.deepEqual(calls, [
    [
      "manage_connector",
      {
        action: "create",
        name: "项目动态",
        runtime_names: ["项目跟进", "notes"],
        binding: { url: "https://example.com/events" },
      },
    ],
  ]);
});

test("the host rejects presentation data, retired actions and malformed source operations", async () => {
  const client = {
    request: () => assert.fail("invalid requests must not reach the backend"),
  } as unknown as ReMeClient;
  const base = {
    type: "connector",
    action: "update",
    source_id: "preset:screen_capture_cron",
  };
  for (const value of [
    base,
    { ...base, enabled: null },
    { ...base, enabled: "false" },
    { ...base, emoji: "📋", enabled: false },
    { ...base, action: "pause" },
    { ...base, connector_job: "other", enabled: false },
    { ...base, source_id: "../other", enabled: false },
    { ...base, name: " " },
    { ...base, runtime_names: ["Notes", "notes"] },
    { ...base, runtime_names: ["../task"] },
    {
      type: "connector",
      action: "create",
      name: "source",
      runtime_names: [],
      binding: { url: "https://example.com" },
    },
    {
      type: "connector",
      action: "create",
      name: "source",
      runtime_names: ["notes"],
      binding: { url: "file:///private/events" },
    },
  ]) {
    await assert.rejects(
      requestWorkspace(
        client,
        value as ConnectorRequest,
        new AbortController().signal,
      ),
      { key: "invalidWorkspaceRequest" },
    );
  }
});
