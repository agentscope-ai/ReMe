import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const manifestUrl = new URL("../.generated/content/manifest.json", import.meta.url);

test("includes the historical Agent integration plan in Chinese navigation", async () => {
  const manifest = JSON.parse(await readFile(manifestUrl, "utf8"));
  const document = manifest.documents.find(
    (item) => item.id === "zh-agent-integration-plan",
  );

  assert.deepEqual(document, {
    id: "zh-agent-integration-plan",
    path: "docs/zh/agent_integration_plan.md",
    sourcePath: "docs/zh/agent_integration_plan.md",
    title: "Agent 集成设计与调研记录",
    description:
      "Codex、DSH、OpenClaw、Claude Code 与 Hermes Agent 的历史设计和调研记录。",
    group: "integration",
    language: "zh",
  });
});
