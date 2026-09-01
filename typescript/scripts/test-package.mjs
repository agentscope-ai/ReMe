import { execFile } from "node:child_process";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
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
      'import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";',
      'assert.equal(typeof ReMeClient, "function");',
      'assert.equal(typeof formatReMeContext, "function");',
      'assert.ok(import.meta.resolve("@agentscope-ai/reme/openclaw").endsWith("/dist/openclaw/index.js"));',
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

  const clawHubOutputDirectory = path.join(temporaryDirectory, "clawhub-pack");
  await mkdir(clawHubOutputDirectory, { recursive: true });
  const packClawHubScript = fileURLToPath(
    new URL("./pack-clawhub.mjs", import.meta.url),
  );
  const clawHubPackResult = await execFileAsync(
    "node",
    [packClawHubScript, clawHubOutputDirectory],
    { cwd: packageDirectory },
  );
  const { tarball: clawHubTarball } = JSON.parse(clawHubPackResult.stdout);
  const clawHubConsumerDirectory = path.join(
    temporaryDirectory,
    "clawhub-consumer",
  );
  await mkdir(clawHubConsumerDirectory, { recursive: true });
  await writeFile(
    path.join(clawHubConsumerDirectory, "package.json"),
    JSON.stringify({ private: true, type: "module" }),
  );
  await execFileAsync(
    "npm",
    ["install", "--ignore-scripts", "--no-audit", "--no-fund", clawHubTarball],
    { cwd: clawHubConsumerDirectory },
  );
  const clawHubPackageDirectory = path.join(
    clawHubConsumerDirectory,
    "node_modules",
    "@agentscope-ai",
    "reme",
  );
  const clawHubReadme = await readFile(
    path.join(clawHubPackageDirectory, "README.md"),
    "utf8",
  );
  const clawHubReadmeZh = await readFile(
    path.join(clawHubPackageDirectory, "README_ZH.md"),
    "utf8",
  );
  assert.match(clawHubReadme, /^# ReMe memory for OpenClaw/m);
  assert.doesNotMatch(clawHubReadme, /DeepSeek Harness/);
  assert.match(clawHubReadmeZh, /^# OpenClaw 的 ReMe 长期记忆插件/m);
  assert.doesNotMatch(clawHubReadmeZh, /DeepSeek Harness/);
  assert.equal(
    JSON.parse(
      await readFile(
        path.join(clawHubPackageDirectory, "package.json"),
        "utf8",
      ),
    ).version,
    sourceManifest.version,
  );
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
