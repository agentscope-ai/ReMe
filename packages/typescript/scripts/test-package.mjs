import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const packageDirectory = new URL("..", import.meta.url);
const sourceManifest = JSON.parse(
  await readFile(new URL("../package.json", import.meta.url), "utf8"),
);
const temporaryDirectory = await mkdtemp(
  path.join(tmpdir(), "reme-package-smoke-"),
);

try {
  const packResult = await execFileAsync(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      temporaryDirectory,
    ],
    { cwd: packageDirectory },
  );
  const [{ filename }] = JSON.parse(packResult.stdout);
  const tarball = path.join(temporaryDirectory, filename);
  const consumerEntry = path.join(temporaryDirectory, "consumer.mjs");
  await writeFile(
    path.join(temporaryDirectory, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  await execFileAsync(
    "npm",
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", tarball],
    { cwd: temporaryDirectory },
  );
  await writeFile(
    consumerEntry,
    [
      'import assert from "node:assert/strict";',
      'import { readFile } from "node:fs/promises";',
      'import plugin from "@agentscope-ai/reme/openclaw";',
      'import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";',
      'assert.equal(typeof ReMeClient, "function");',
      'assert.equal(typeof formatReMeContext, "function");',
      'assert.equal(plugin.id, "reme");',
      'assert.match(import.meta.resolve("@agentscope-ai/reme/dsh"), /dist\\/dsh\\/index\\.js$/);',
      'assert.match(import.meta.resolve("@agentscope-ai/reme/client"), /dist\\/dsh\\/client\\.js$/);',
      'const manifestUrl = import.meta.resolve("@agentscope-ai/reme/package.json");',
      'const manifest = JSON.parse(await readFile(new URL(manifestUrl), "utf8"));',
      `assert.equal(manifest.version, ${JSON.stringify(
        sourceManifest.version,
      )});`,
      'assert.match(await readFile(new URL("README_ZH.md", manifestUrl), "utf8"), /TypeScript Agent/);',
    ].join("\n"),
  );
  await execFileAsync("node", [consumerEntry], { cwd: temporaryDirectory });
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
