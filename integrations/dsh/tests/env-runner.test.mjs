import assert from "node:assert/strict";
import test from "node:test";
import {
  closeSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createInstallSpawner } from "../dist/environment/host/install.js";
import {
  openAppendLog,
  pruneInstallLogs,
} from "../dist/environment/host/log.js";
import { createNodeRunner } from "../dist/environment/host/runner.js";

test("a timeout settles with an explicit error instead of hanging", async () => {
  // The child ignores SIGTERM, so the grace path must force it and settle.
  const run = createNodeRunner({ PATH: process.env.PATH });
  const started = Date.now();
  const result = await run(
    process.execPath,
    ["-e", "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"],
    { timeoutMs: 200 },
  );
  assert.match(result.error, /timed out after 200 ms/);
  assert.ok(
    Date.now() - started < 5_000,
    "the probe must not wait for a process that never closes",
  );
});

test("a child that exits on SIGTERM is still reported as a timeout", async () => {
  const run = createNodeRunner({ PATH: process.env.PATH });
  const result = await run(
    process.execPath,
    ["-e", "setInterval(() => {}, 1000)"],
    { timeoutMs: 200 },
  );
  assert.match(result.error ?? "", /timed out after 200 ms/);
});

test("the real install spawner starts and stops a process", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-spawner-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const spawner = createInstallSpawner({ PATH: process.env.PATH });
  const logPath = join(directory, "install.log");
  const child = spawner.start({
    exe: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    logPath,
  });
  assert.ok(child.pid !== undefined && child.pid > 1);

  const exited = new Promise((resolve) => child.onExit(resolve));
  child.kill();
  const outcome = await Promise.race([
    exited,
    new Promise((resolve) => setTimeout(() => resolve("timeout"), 5_000)),
  ]);
  assert.notEqual(outcome, "timeout", "the install process must actually exit");
  assert.match(readFileSync(logPath, "utf8"), /^$/);
});

test("an oversized log is rotated before the next append", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-log-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, "install.log");
  writeFileSync(path, "x".repeat(200));

  const fd = openAppendLog(path, 100);
  closeSync(fd);
  assert.equal(readFileSync(`${path}.1`, "utf8").length, 200);
  assert.equal(statSync(path).size, 0);
});

test("per-job install logs are pruned to the newest files", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-prune-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  for (let index = 0; index < 25; index += 1)
    writeFileSync(join(directory, `install-${index}.log`), "x");

  pruneInstallLogs(directory, 20);
  const remaining = readdirSync(directory).filter((name) =>
    name.endsWith(".log"),
  );
  assert.equal(remaining.length, 20);
});
