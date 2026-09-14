import { execFile } from "node:child_process";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
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
} finally {
  await rm(temporaryDirectory, { force: true, recursive: true });
}
