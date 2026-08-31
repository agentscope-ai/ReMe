import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const manifestUrl = new URL("../.generated/content/manifest.json", import.meta.url);

test("omits retired Agent documents and places Agent integration after getting started", async () => {
  const manifest = JSON.parse(await readFile(manifestUrl, "utf8"));
  const documents = manifest.documents;
  const groups = [...new Set(documents.map((document) => document.group))];

  assert.equal(documents.some((document) => document.id === "reme-memory-skill"), false);
  assert.equal(documents.some((document) => document.sourcePath.endsWith("agent_integration_plan.md")), false);
  assert.deepEqual(
    documents
      .filter((document) => document.group === "plugins")
      .map((document) => document.title || document.titles?.en),
    ["每日论文插件", "Auto Fin 插件", "Daily Paper Plugin", "Auto Fin Plugin"],
  );
  assert.equal(documents.some((document) => document.group === "cookbooks"), false);
  assert.ok(groups.indexOf("integration") > groups.indexOf("start"));
  assert.ok(groups.indexOf("integration") < groups.indexOf("fundamentals"));
});
