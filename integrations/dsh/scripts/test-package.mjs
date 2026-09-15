import { execFile } from "node:child_process";
import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
const temporaryDirectory = await mkdtemp(
  path.join(tmpdir(), "reme-dsh-package-"),
);

try {
  const { stdout } = await execFileAsync(
    "npm",
    [
      "pack",
      "--json",
      "--ignore-scripts",
      "--pack-destination",
      temporaryDirectory,
    ],
    { cwd: new URL("..", import.meta.url) },
  );
  const [result] = JSON.parse(stdout);
  const files = new Set(result.files.map(({ path: file }) => file));
  for (const file of [
    "dist/index.js",
    "dist/client.js",
    "cordis.patch.yml",
    "README.md",
    "figures/reme-status-overview.png",
  ]) {
    assert.ok(files.has(file), `missing ${file}`);
  }
  assert.ok(![...files].some((file) => file.includes("openclaw")));

  const consumerDirectory = path.join(temporaryDirectory, "consumer");
  await mkdir(consumerDirectory);
  await writeFile(
    path.join(consumerDirectory, "package.json"),
    '{"name":"reme-dsh-package-consumer","private":true}\n',
  );
  const hostDependencies = [
    "@deepseek-ai/dsh-client-ui-primitives",
    "@deepseek-ai/dsh-llm",
    "@deepseek-ai/dsh-settings",
    "@deepseek-ai/dsh-tools",
    "@deepseek-ai/dsh-typert-protocol",
  ].map((dependency) => `${dependency}@0.1.5-rc.2`);
  await execFileAsync(
    "npm",
    [
      "install",
      "--ignore-scripts",
      "--no-audit",
      "--no-fund",
      path.join(temporaryDirectory, result.filename),
      ...hostDependencies,
    ],
    { cwd: consumerDirectory },
  );
  await execFileAsync(
    process.execPath,
    [
      "--input-type=module",
      "--eval",
      [
        'const plugin = await import("@agentscope-ai/reme-dsh-plugin");',
        'if (typeof plugin.apply !== "function") throw new Error("missing apply export");',
        'if (!Array.isArray(plugin.inject)) throw new Error("missing inject export");',
        'if (plugin.Config === undefined) throw new Error("missing Config export");',
      ].join("\n"),
    ],
    { cwd: consumerDirectory },
  );
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
