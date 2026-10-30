import assert from "node:assert/strict";
import test from "node:test";
import { createServer } from "node:http";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  buildStartCommand,
  createServiceIO,
  forgetService,
  ownsService,
  parseLocalEndpoint,
  readServiceState,
  serviceOwnership,
  serviceStatus,
  startService,
  stopService,
  writeServiceState,
} from "../dist/environment/host/service.js";

const ENDPOINT = "http://127.0.0.1:2334";

function fakeIO(overrides = {}) {
  const calls = { killTree: [], force: [], spawn: [] };
  const io = {
    fetchHealth: async () => ({ ok: false, version: "", error: "" }),
    portOpen: async () => false,
    async spawnService(options) {
      calls.spawn.push(options);
      return { pid: 4321 };
    },
    killTree(pid, force = false) {
      calls.killTree.push(pid);
      if (force) calls.force.push(pid);
    },
    probePid() {},
    processImage: async () => "python.exe",
    processCommandLine: async () => "",
    exists: () => true,
    platform: "win32",
    ...overrides,
  };
  return { io, calls };
}

/** The command line a launch for `statePath` would actually have. */
function commandFor(statePath, port = 2334) {
  const state = readServiceState(statePath);
  return `reme start workspace_dir=${state.workspaceDir} service.port=${port}`;
}

function tempState(t, prefix) {
  const directory = mkdtempSync(join(tmpdir(), prefix));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  return { directory, statePath: join(directory, "service.json") };
}

function record(statePath, overrides = {}) {
  writeServiceState(statePath, {
    pid: 777,
    endpoint: ENDPOINT,
    workspaceDir: join(statePath, ".."),
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
    ...overrides,
  });
}

test("accepts only loopback HTTP endpoints and validates the port", () => {
  assert.deepEqual(parseLocalEndpoint("http://127.0.0.1:2334"), {
    host: "127.0.0.1",
    port: 2334,
  });
  assert.deepEqual(parseLocalEndpoint("http://localhost:2333"), {
    host: "127.0.0.1",
    port: 2333,
  });
  assert.throws(() => parseLocalEndpoint("http://10.0.0.5:2333"));
  assert.throws(() => parseLocalEndpoint("not-a-url"));
  assert.throws(() => parseLocalEndpoint("http://127.0.0.1:0"));
  // The plugin starts a plain HTTP service, so https on loopback can never be
  // healthy; it must not be offered as manageable.
  assert.throws(
    () => parseLocalEndpoint("https://127.0.0.1:2333"),
    /http:\/\//,
  );
  // The launch binds service.host=127.0.0.1, so an IPv6 loopback endpoint
  // would start a service that can never answer it.
  assert.throws(() => parseLocalEndpoint("http://[::1]:2333"), /IPv4/);
});

test("builds reme start with the workspace cwd contract", () => {
  const command = buildStartCommand({
    cliPath: "D:\\env\\Scripts\\reme.exe",
    pythonPath: "D:\\env\\python.exe",
    workspaceDir: "C:\\Users\\u\\.reme",
    endpoint: "http://127.0.0.1:2334",
    disableDreamCron: false,
    exists: () => true,
  });
  assert.equal(command.exe, "D:\\env\\Scripts\\reme.exe");
  assert.ok(command.args.includes("start"));
  assert.ok(command.args.includes("workspace_dir=C:\\Users\\u\\.reme"));
  assert.ok(command.args.includes("service.host=127.0.0.1"));
  assert.ok(command.args.includes("service.port=2334"));
  assert.ok(command.args.includes("enable_logo=false"));
  assert.ok(!command.args.includes("jobs.dream_cron.enabled=false"));

  const newer = buildStartCommand({
    cliPath: "D:\\env\\Scripts\\reme.exe",
    pythonPath: "D:\\env\\python.exe",
    workspaceDir: "C:\\Users\\u\\.reme",
    endpoint: "http://127.0.0.1:2334",
    disableDreamCron: true,
    exists: () => true,
  });
  assert.ok(newer.args.includes("jobs.dream_cron.enabled=false"));
});

test("falls back to the interpreter wrapper when the CLI is missing", () => {
  const command = buildStartCommand({
    cliPath: "",
    pythonPath: "D:\\env\\python.exe",
    workspaceDir: "C:\\Users\\u\\.reme",
    endpoint: "http://127.0.0.1:2334",
    disableDreamCron: false,
    exists: () => false,
  });
  assert.equal(command.exe, "D:\\env\\python.exe");
  assert.equal(command.args[0], "-c");
  assert.match(String(command.args[1]), /sys\.argv/);
  assert.match(String(command.args[1]), /from reme\.reme import main/);
  assert.match(String(command.args[1]), /service\.port=2334/);
});

test("derives running, conflict and stopped phases", async (t) => {
  const { directory, statePath } = tempState(t, "reme-service-");
  const base = {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  };

  const runningIo = fakeIO({
    fetchHealth: async () => ({ ok: true, version: "0.4.0.9", error: "" }),
  });
  let status = await serviceStatus(runningIo.io, base);
  assert.equal(status.phase, "running");
  assert.equal(status.managed, false);
  assert.equal(status.version, "0.4.0.9");

  record(statePath);
  runningIo.io.processCommandLine = async () => commandFor(statePath);
  status = await serviceStatus(runningIo.io, base);
  assert.equal(status.phase, "running");
  assert.equal(status.managed, true);
  assert.equal(status.pid, 777);

  const conflict = await serviceStatus(
    fakeIO({ portOpen: async () => true }).io,
    base,
  );
  assert.equal(conflict.phase, "conflict");

  const stopped = await serviceStatus(fakeIO().io, base);
  assert.equal(stopped.phase, "stopped");
});

test("a service on another endpoint is not reported as managed", async (t) => {
  const { directory, statePath } = tempState(t, "reme-own-");
  record(statePath, { endpoint: "http://127.0.0.1:2333" });
  const { io } = fakeIO({
    fetchHealth: async () => ({ ok: true, version: "0.4.0.9", error: "" }),
  });
  const status = await serviceStatus(io, {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  });
  assert.equal(status.phase, "running");
  assert.equal(status.managed, false);
  assert.equal(status.pid, undefined);
});

test("a loopback alias still identifies the service this plugin started", async (t) => {
  const { directory, statePath } = tempState(t, "reme-alias-");
  record(statePath, { endpoint: "http://localhost:2334/" });
  const { io } = fakeIO({
    fetchHealth: async () => ({ ok: true, version: "0.4.0.9", error: "" }),
    processCommandLine: async () => commandFor(statePath),
  });
  const status = await serviceStatus(io, {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  });
  assert.equal(status.managed, true);
  assert.equal(status.pid, 777);
});

test("a conflict still identifies a provably owned service", async (t) => {
  const { directory, statePath } = tempState(t, "reme-conflict-");
  record(statePath);
  const { io } = fakeIO({
    portOpen: async () => true,
    processCommandLine: async () => commandFor(statePath),
  });
  const status = await serviceStatus(io, {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  });
  assert.equal(status.phase, "conflict");
  assert.equal(status.managed, true);
  assert.equal(status.pid, 777);
});

test("a recycled pid that is no longer the service is not managed", async (t) => {
  const { statePath } = tempState(t, "reme-pid-");
  record(statePath);
  const { io } = fakeIO({
    fetchHealth: async () => ({ ok: true, version: "0.4.0.9", error: "" }),
    processImage: async () => "node.exe",
  });
  assert.equal(
    await ownsService(io, readServiceState(statePath), ENDPOINT),
    false,
  );
  const { io: gone } = fakeIO({
    probePid() {
      throw new Error("ESRCH");
    },
  });
  assert.equal(
    await ownsService(gone, readServiceState(statePath), ENDPOINT),
    false,
  );
  // Ownership guards a destructive action: an image that cannot be read is
  // not evidence that the pid is ours, and a prefix match would accept e.g.
  // pycharm64.exe.
  for (const image of ["", "   ", "pycharm64.exe", "python-config"]) {
    const { io: unreadable } = fakeIO({ processImage: async () => image });
    assert.equal(
      await ownsService(unreadable, readServiceState(statePath), ENDPOINT),
      false,
      `image ${JSON.stringify(image)} must not be owned`,
    );
  }
  // A Python image whose command line describes another launch is foreign,
  // not ours; an unreadable command line is unproven, not ownership.
  const { io: otherLaunch } = fakeIO({
    processCommandLine: async () => "python -m http.server 2334",
  });
  assert.equal(
    await serviceOwnership(otherLaunch, readServiceState(statePath), ENDPOINT),
    "foreign",
  );
  const { io: unproven } = fakeIO();
  assert.equal(
    await serviceOwnership(unproven, readServiceState(statePath), ENDPOINT),
    "unknown",
  );
  assert.equal(
    await ownsService(unproven, readServiceState(statePath), ENDPOINT),
    false,
  );
  // Matching is boundary-aware: a longer workspace path or port is another
  // launch, not this one. A path separator is not a boundary either, so a
  // record for `...\reme-x` never claims a service started in `...\reme-x\sub`.
  const state = readServiceState(statePath);
  for (const command of [
    `reme start workspace_dir=${state.workspaceDir}2 service.port=2334`,
    `reme start workspace_dir=${state.workspaceDir}\\sub service.port=2334`,
    `reme start workspace_dir=${state.workspaceDir}/sub service.port=2334`,
    `reme start workspace_dir=${state.workspaceDir} service.port=23340`,
  ]) {
    const { io: longer } = fakeIO({
      processCommandLine: async () => command,
    });
    assert.equal(
      await serviceOwnership(longer, state, ENDPOINT),
      "foreign",
      command,
    );
  }
});

test("recognizes the Windows wrapper command line with escaped quotes", async (t) => {
  const { statePath } = tempState(t, "reme-wrapper-");
  record(statePath);
  const state = readServiceState(statePath);
  // What Windows shows for `python -c <wrapper>`: the JSON quotes are escaped
  // as `\"` and each path separator appears doubled.
  const escaped = state.workspaceDir.replaceAll("\\", "\\\\");
  const command = `python -c "import sys;sys.argv=[\\"reme\\",\\"start\\",\\"workspace_dir=${escaped}\\",\\"service.port=2334\\"];from reme.reme import main"`;
  const { io } = fakeIO({ processCommandLine: async () => command });
  assert.equal(await serviceOwnership(io, state, ENDPOINT), "owned");
});

test("a dead record is cleared instead of reporting a stale pid", async (t) => {
  const { directory, statePath } = tempState(t, "reme-dead-");
  record(statePath);
  const { io } = fakeIO({
    probePid() {
      throw new Error("ESRCH");
    },
  });
  const status = await serviceStatus(io, {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  });
  assert.equal(status.phase, "stopped");
  assert.equal(status.pid, undefined);
  assert.equal(readServiceState(statePath), undefined);
});

test("a probe denied by permissions is not treated as a dead process", async (t) => {
  const { directory, statePath } = tempState(t, "reme-eperm-");
  record(statePath);
  const denied = new Error("operation not permitted");
  denied.code = "EPERM";
  const { io } = fakeIO({
    probePid() {
      throw denied;
    },
  });
  const status = await serviceStatus(io, {
    endpoint: ENDPOINT,
    statePath,
    logPath: join(directory, "service.log"),
  });
  // The process exists but this user cannot signal it (e.g. an elevated
  // service): the record must survive so it is not orphaned.
  assert.equal(status.phase, "stopped");
  assert.equal(status.pid, 777);
  assert.equal(readServiceState(statePath)?.pid, 777);
});

test("starts detached with the GBK-safe environment and records state", async (t) => {
  const { directory, statePath } = tempState(t, "reme-start-");
  const { io, calls } = fakeIO();
  const result = await startService(io, {
    endpoint: ENDPOINT,
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    cliPath: "D:\\env\\Scripts\\reme.exe",
    disableDreamCron: true,
    statePath,
    logPath: join(directory, "service.log"),
    env: { PATH: "x" },
  });
  assert.equal(result.started, true);
  assert.equal(result.pid, 4321);
  const spawned = calls.spawn[0];
  assert.equal(spawned.cwd, directory);
  assert.equal(spawned.env.PYTHONUTF8, "1");
  assert.equal(spawned.env.NO_PROXY, "localhost,127.0.0.1,::1");
  const persisted = readServiceState(statePath);
  assert.equal(persisted?.pid, 4321);
  assert.equal(persisted?.workspaceDir, directory);
});

test("records nothing when the spawn fails", async (t) => {
  const { directory, statePath } = tempState(t, "reme-spawn-");
  const { io } = fakeIO({
    async spawnService() {
      throw new Error("spawn ENOENT");
    },
  });
  const result = await startService(io, {
    endpoint: ENDPOINT,
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    cliPath: "D:\\missing\\reme.exe",
    disableDreamCron: false,
    statePath,
    logPath: join(directory, "service.log"),
    env: {},
  });
  assert.equal(result.started, false);
  assert.match(result.error, /ENOENT/);
  assert.equal(readServiceState(statePath), undefined);
});

test("stops the managed pid and clears the state file after it exits", async (t) => {
  const { directory, statePath } = tempState(t, "reme-stop-");
  record(statePath, { pid: 999, workspaceDir: directory });
  const { io, calls } = fakeIO({
    processCommandLine: async () => commandFor(statePath),
  });
  let running = true;
  io.probePid = () => {
    if (!running) throw new Error("ESRCH");
  };
  io.killTree = (pid) => {
    calls.killTree.push(pid);
    running = false;
  };
  const stopped = await stopService(io, {
    statePath,
    endpoint: ENDPOINT,
    timeoutMs: 100,
    pollMs: 5,
  });
  assert.deepEqual(stopped, { stopped: true, pid: 999 });
  assert.deepEqual(calls.killTree, [999]);
  assert.equal(readServiceState(statePath), undefined);
  assert.deepEqual(await stopService(io, { statePath, endpoint: ENDPOINT }), {
    stopped: false,
    reason: "No service started by this plugin",
    code: "no-record",
  });
});

test("escalates to SIGKILL and keeps the record when the service never exits", async (t) => {
  const { directory, statePath } = tempState(t, "reme-stubborn-");
  record(statePath, { pid: 999, workspaceDir: directory });
  const { io, calls } = fakeIO({
    processCommandLine: async () => commandFor(statePath),
  });
  const stopped = await stopService(io, {
    statePath,
    endpoint: ENDPOINT,
    timeoutMs: 10,
    forceTimeoutMs: 10,
    pollMs: 2,
  });
  assert.equal(stopped.stopped, false);
  assert.match(stopped.reason, /did not exit/);
  // The record survives, so it is named: the tab can offer to forget exactly
  // this record.
  assert.deepEqual(stopped.record, { endpoint: ENDPOINT, pid: 999 });
  assert.deepEqual(calls.killTree, [999, 999]);
  assert.deepEqual(calls.force, [999]);
  // The pid survives in the record, so Stop can be tried again later.
  assert.equal(readServiceState(statePath)?.pid, 999);
});

test("refuses to stop a service recorded for another endpoint", async (t) => {
  const { statePath } = tempState(t, "reme-refuse-");
  record(statePath, { endpoint: "http://127.0.0.1:2333" });
  const { io, calls } = fakeIO();
  const stopped = await stopService(io, { statePath, endpoint: ENDPOINT });
  assert.equal(stopped.stopped, false);
  assert.match(stopped.reason, /does not belong/);
  assert.deepEqual(calls.killTree, []);
  assert.ok(readServiceState(statePath), "the foreign record is left in place");
});

test("refuses to stop a process whose identity cannot be verified", async (t) => {
  const { statePath } = tempState(t, "reme-unknown-");
  record(statePath);
  const { io, calls } = fakeIO(); // image readable, command line not
  const stopped = await stopService(io, { statePath, endpoint: ENDPOINT });
  assert.equal(stopped.stopped, false);
  assert.match(stopped.reason, /could not be verified/);
  // The record is named, so the client can offer the record-only way out.
  assert.deepEqual(stopped.record, { endpoint: ENDPOINT, pid: 777 });
  assert.deepEqual(calls.killTree, []);
  assert.equal(readServiceState(statePath)?.pid, 777);
});

test("forgetting refuses a record that changed since it was inspected", async (t) => {
  const { statePath } = tempState(t, "reme-changed-");
  record(statePath, { pid: 999 });
  const { calls } = fakeIO();
  const changed = await forgetService({
    statePath,
    endpoint: ENDPOINT,
    expectPid: 777,
  });
  assert.equal(changed.forgotten, false);
  assert.equal(changed.code, "record-changed");
  assert.deepEqual(changed.record, { endpoint: ENDPOINT, pid: 999 });
  // The record is still there: dropping it would orphan the process.
  assert.equal(readServiceState(statePath)?.pid, 999);
  assert.deepEqual(calls.killTree, []);

  const forgotten = await forgetService({
    statePath,
    endpoint: ENDPOINT,
    expectPid: 999,
  });
  assert.equal(forgotten.forgotten, true);
  assert.equal(readServiceState(statePath), undefined);
});

/* ------------------------------------------------------------------ */
/* Real IO: the pieces the fake adapter above cannot vouch for.       */
/* ------------------------------------------------------------------ */

async function listening(handler, t) {
  const server = createServer(handler);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    // fetch keeps pooled sockets alive; close them so the next test is not
    // racing a server that is still draining.
    server.closeAllConnections?.();
    server.close();
  });
  const address = server.address();
  return `http://127.0.0.1:${address.port}`;
}

/**
 * Windows may keep a just-killed process's cwd/log handle for a moment, so an
 * immediate recursive remove can fail with EBUSY and fail the whole test.
 * Only EBUSY/ENOTEMPTY are transient; anything else (and a persistent lock)
 * fails loudly instead of leaking a temporary directory.
 */
async function removeDirectory(path) {
  for (let attempt = 0; ; attempt += 1) {
    try {
      rmSync(path, { recursive: true, force: true });
      return;
    } catch (error) {
      const code = error?.code;
      if ((code !== "EBUSY" && code !== "ENOTEMPTY") || attempt >= 19)
        throw error;
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }
}

test("real IO accepts only a ReMe health report", async (t) => {
  // One server for every case: several servers plus the 3 s probe timeout made
  // this test fail under parallel load.
  let authorization = "";
  const base = await listening((request, response) => {
    response.setHeader("Content-Type", "application/json");
    const path = String(request.url ?? "").replace(/\/health_check$/, "");
    if (path === "/reme")
      response.end(
        JSON.stringify({
          success: true,
          metadata: {
            health: { version: "0.4.2.1", healthy: true, components: {} },
          },
        }),
      );
    else if (path === "/partial")
      // A health object that does not parse is not a ReMe health report: the
      // shared client requires the version, the health flag and the
      // components, so a generic `{}` never passes.
      response.end(
        JSON.stringify({
          success: true,
          metadata: { health: { version: "0.4.2.1" } },
        }),
      );
    else if (path === "/failing") {
      response.statusCode = 503;
      response.end("{}");
    } else if (path === "/unhappy")
      response.end(JSON.stringify({ success: false }));
    else if (path === "/guarded") {
      authorization = String(request.headers.authorization ?? "");
      response.end("{}");
    } else response.end("{}");
  }, t);

  const io = createServiceIO();
  const good = await io.fetchHealth(`${base}/reme`, "");
  assert.equal(good.ok, true);
  assert.equal(good.version, "0.4.2.1");
  assert.equal(good.code, undefined);

  const empty = await io.fetchHealth(`${base}/generic`, "");
  assert.equal(empty.ok, false);
  assert.match(empty.error, /ReMe health report/);
  // The client localizes the code instead of showing this English sentence.
  assert.equal(empty.code, "not-reme");

  // A health report that does not carry the expected fields is not one.
  const partial = await io.fetchHealth(`${base}/partial`, "");
  assert.equal(partial.ok, false);
  assert.equal(partial.code, "not-reme");

  const httpError = await io.fetchHealth(`${base}/failing`, "");
  assert.equal(httpError.code, "http-status");

  const reported = await io.fetchHealth(`${base}/unhappy`, "");
  assert.equal(reported.code, "reme-error");

  // The Authorization header is actually sent when a token is configured.
  await io.fetchHealth(`${base}/guarded`, "secret-token");
  assert.equal(authorization, "Bearer secret-token");
});

test("real IO spawns, identifies and stops a detached process", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "reme-real-service-"));
  t.after(() => removeDirectory(directory));
  const io = createServiceIO();
  const { pid } = await io.spawnService({
    exe: process.execPath,
    args: ["-e", "setInterval(() => {}, 1000)"],
    cwd: directory,
    logPath: join(directory, "service.log"),
    env: process.env,
  });
  assert.ok(pid !== undefined && pid > 1);
  assert.doesNotThrow(() => io.probePid(pid));
  assert.doesNotMatch(await io.processImage(pid), /^$/);
  assert.match(await io.processCommandLine(pid), /node/i);

  io.killTree(pid);
  const deadline = Date.now() + 5_000;
  let running = true;
  while (running && Date.now() < deadline) {
    try {
      io.probePid(pid);
      await new Promise((resolve) => setTimeout(resolve, 50));
    } catch {
      running = false;
    }
  }
  assert.equal(running, false, "the detached process must actually exit");
});
