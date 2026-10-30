import assert from "node:assert/strict";
import test from "node:test";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ReMeEnvManager } from "../dist/environment/host/manager.js";
import {
  readServiceState,
  writeServiceState,
} from "../dist/environment/host/service.js";

/** A service adapter that records launches instead of starting processes. */
function fakeServiceIO() {
  const calls = { killTree: [], spawn: [] };
  return {
    calls,
    io: {
      fetchHealth: async () => ({ ok: false, version: "", error: "" }),
      portOpen: async () => false,
      async spawnService(options) {
        calls.spawn.push(options);
        // A launch is only visible once the OS acknowledged it.
        await new Promise((resolve) => setTimeout(resolve, 10));
        return { pid: 5000 + calls.spawn.length };
      },
      killTree() {},
      probePid() {},
      processImage: async () => "python.exe",
      processCommandLine: async () => "",
      exists: () => true,
      platform: "win32",
    },
  };
}

function managerIn(t) {
  const directory = mkdtempSync(join(tmpdir(), "reme-manager-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const { io, calls } = fakeServiceIO();
  const manager = new ReMeEnvManager({ DSH_HOME: directory }, io);
  // Both probes are Host process work; the test replaces them with facts.
  manager.checkReme = async () => ({
    installed: true,
    version: "0.4.2.1",
    versionOk: true,
    hasCore: true,
    cliPath: "",
    error: "",
  });
  manager.serviceStatus = async (endpoint) => ({
    phase: "stopped",
    managed: false,
    endpoint: endpoint || "http://127.0.0.1:2333",
    logPath: join(directory, "service.log"),
  });
  return { manager, calls, directory, io };
}

test("serializes concurrent starts into one service launch", async (t) => {
  const { manager, calls, directory } = managerIn(t);
  const options = [
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  ];
  const [first, second] = await Promise.all([
    manager.serviceStart(...options),
    manager.serviceStart(...options),
  ]);
  assert.equal(calls.spawn.length, 1, "the second request joins the first");
  assert.equal(first.started, true);
  assert.equal(second.started, false);
  assert.equal(second.alreadyRunning, true);
});

test("a failed spawn can be retried immediately", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  io.spawnService = async (options) => {
    calls.spawn.push(options);
    if (calls.spawn.length === 1) throw new Error("spawn ENOENT");
    return { pid: 7000 };
  };
  const options = [
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  ];
  const first = await manager.serviceStart(...options);
  assert.equal(first.started, false);
  assert.match(first.error, /ENOENT/);

  // The start marker was cleared with the failure; a retry must really spawn
  // instead of being reported as "already starting".
  const second = await manager.serviceStart(...options);
  assert.equal(second.started, true);
  assert.equal(calls.spawn.length, 2);
});

test("refuses to spawn while another process owns the port", async (t) => {
  const { manager, calls, directory } = managerIn(t);
  manager.serviceStatus = async (endpoint) => ({
    phase: "conflict",
    managed: false,
    endpoint,
    logPath: join(directory, "service.log"),
  });
  const result = await manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(result.started, false);
  assert.match(result.error, /already listening/);
  assert.equal(calls.spawn.length, 0);
});

test("stops a previous launch that never became healthy before replacing it", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  manager.serviceStatus = async (endpoint) => ({
    phase: "stopped",
    managed: false,
    endpoint,
    // A live record for this endpoint means a launch that is still booting.
    pid: 6000,
    logPath: join(directory, "service.log"),
  });
  writeServiceState(join(directory, "reme-memory", "service.json"), {
    pid: 6000,
    endpoint: "http://127.0.0.1:2334",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });
  let running = true;
  io.probePid = () => {
    if (!running) throw new Error("ESRCH");
  };
  io.killTree = (pid) => {
    calls.killTree.push(pid);
    running = false;
  };
  io.processCommandLine = async () =>
    `reme start workspace_dir=${directory} service.port=2334`;

  const result = await manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(result.started, true);
  assert.deepEqual(calls.killTree, [6000]);
  assert.equal(calls.spawn.length, 1);
});

test("an unprovable previous record blocks the start instead of orphaning it", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  manager.serviceStatus = async (endpoint) => ({
    phase: "stopped",
    managed: false,
    endpoint,
    pid: 6000,
    logPath: join(directory, "service.log"),
  });
  const statePath = join(directory, "reme-memory", "service.json");
  writeServiceState(statePath, {
    pid: 6000,
    endpoint: "http://127.0.0.1:2334",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });
  // The process exists, but its command line cannot be read, so ownership is
  // unproven. Dropping the record would orphan it: it must stay, and the start
  // must report why instead of spawning a second service.
  io.processCommandLine = async () => "";
  const result = await manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(result.started, false);
  assert.equal(result.code, "unverified-record");
  assert.equal(result.pid, 6000, "the client needs the pid in the message");
  // The record is named, so the client can offer to forget exactly this one
  // instead of the endpoint it happens to be showing.
  assert.deepEqual(result.record, {
    endpoint: "http://127.0.0.1:2334",
    pid: 6000,
  });
  assert.deepEqual(calls.killTree, []);
  assert.equal(calls.spawn.length, 0);
  // Once the evidence is readable again, the same record can be stopped.
  assert.equal(readServiceState(statePath)?.pid, 6000);
});

test("a provably foreign previous record is discarded and replaced", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  manager.serviceStatus = async (endpoint) => ({
    phase: "stopped",
    managed: false,
    endpoint,
    pid: 6000,
    logPath: join(directory, "service.log"),
  });
  const statePath = join(directory, "reme-memory", "service.json");
  writeServiceState(statePath, {
    pid: 6000,
    endpoint: "http://127.0.0.1:2334",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });
  // The live pid belongs to something else now, so the record is provably
  // stale: the start drops it without killing the process and spawns fresh.
  io.processCommandLine = async () => "python -m http.server 2334";
  const result = await manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(result.started, true);
  assert.deepEqual(calls.killTree, [], "a foreign process is never signalled");
  assert.equal(calls.spawn.length, 1);
  assert.equal(readServiceState(statePath)?.pid, 5001);
});

test("forgetting a record removes it without signalling the process", async (t) => {
  const { manager, calls, directory } = managerIn(t);
  const statePath = join(directory, "reme-memory", "service.json");
  writeServiceState(statePath, {
    pid: 6000,
    endpoint: "http://127.0.0.1:2334",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });

  // Another endpoint cannot drop this record.
  const foreign = await manager.serviceForget("http://127.0.0.1:2333");
  assert.deepEqual(foreign, {
    forgotten: false,
    reason: "The recorded service does not belong to this endpoint",
    code: "foreign",
  });
  assert.equal(readServiceState(statePath)?.pid, 6000);

  const forgotten = await manager.serviceForget("http://127.0.0.1:2334");
  assert.deepEqual(forgotten, { forgotten: true, pid: 6000 });
  assert.equal(readServiceState(statePath), undefined);
  // Forgetting only touches the record: the process is never signalled.
  assert.deepEqual(calls.killTree, []);

  const again = await manager.serviceForget("http://127.0.0.1:2334");
  assert.equal(again.forgotten, false);
  assert.equal(again.code, "no-record");
});

test("refuses to forget a record that changed since it was inspected", async (t) => {
  const { manager, directory } = managerIn(t);
  const statePath = join(directory, "reme-memory", "service.json");
  writeServiceState(statePath, {
    pid: 6000,
    endpoint: "http://127.0.0.1:2334",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });

  // The user was shown pid 6000, but the record now describes another pid.
  const changed = await manager.serviceForget("http://127.0.0.1:2334", 7000);
  assert.equal(changed.forgotten, false);
  assert.equal(changed.code, "record-changed");
  assert.equal(readServiceState(statePath)?.pid, 6000, "the record survives");

  // The record the Host names is the one a retry may act on.
  const current = await manager.serviceForget("http://127.0.0.1:2334", 6000);
  assert.equal(current.forgotten, true);
  assert.equal(readServiceState(statePath), undefined);
});

test("a blocked start names the record, and forgetting targets that record", async (t) => {
  // The reported case: a live but unverifiable record for port 2333 while the
  // configured endpoint is 2444. The start is blocked by the 2333 record, so
  // the recovery has to act on that record, not on the endpoint on screen.
  const { manager, directory, io } = managerIn(t);
  const statePath = join(directory, "reme-memory", "service.json");
  writeServiceState(statePath, {
    pid: 6000,
    endpoint: "http://127.0.0.1:2333",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });
  // The process exists, but its command line cannot be read, so ownership is
  // unproven and the record must survive.
  io.processCommandLine = async () => "";

  const started = await manager.serviceStart(
    "http://127.0.0.1:2444",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(started.started, false);
  assert.equal(started.code, "unverified-record");
  assert.deepEqual(started.record, {
    endpoint: "http://127.0.0.1:2333",
    pid: 6000,
  });

  // Forgetting the endpoint on screen is refused as foreign: the record
  // belongs to another port.
  const foreign = await manager.serviceForget("http://127.0.0.1:2444");
  assert.equal(foreign.forgotten, false);
  assert.equal(foreign.code, "foreign");

  // The record's own endpoint, pinned by the pid the Host reported, is what
  // the confirmed recovery targets.
  const forgotten = await manager.serviceForget("http://127.0.0.1:2333", 6000);
  assert.equal(forgotten.forgotten, true);
  assert.equal(readServiceState(statePath), undefined);
});

test("refuses to overwrite a live record owned by another endpoint", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  manager.serviceStatus = async (endpoint) => ({
    phase: "stopped",
    managed: false,
    endpoint,
    logPath: join(directory, "service.log"),
  });
  writeServiceState(join(directory, "reme-memory", "service.json"), {
    pid: 6000,
    endpoint: "http://127.0.0.1:2333",
    workspaceDir: directory,
    pythonPath: "D:\\env\\python.exe",
    command: "reme start",
    startedAt: "2026-10-27T00:00:00.000Z",
  });
  io.processCommandLine = async () =>
    `reme start workspace_dir=${directory} service.port=2333`;
  const result = await manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  assert.equal(result.started, false);
  assert.match(result.error, /still running/);
  assert.equal(calls.spawn.length, 0);
});

test("stop waits for an in-flight start and then stops that launch", async (t) => {
  const { manager, calls, directory, io } = managerIn(t);
  let running = true;
  io.killTree = (pid) => {
    calls.killTree.push(pid);
    running = false;
  };
  io.probePid = () => {
    if (!running) throw new Error("ESRCH");
  };
  io.processCommandLine = async () =>
    `reme start workspace_dir=${join(
      directory,
      "workspace",
    )} service.port=2334`;

  const start = manager.serviceStart(
    "http://127.0.0.1:2334",
    "D:\\env\\python.exe",
    join(directory, "workspace"),
  );
  // Arrives while the start is still being recorded, so it must join the queue
  // rather than answering "No service started by this plugin".
  const stop = manager.serviceStop("http://127.0.0.1:2334");
  const [started, stopped] = await Promise.all([start, stop]);
  assert.equal(started.started, true);
  assert.equal(stopped.stopped, true);
  assert.deepEqual(calls.killTree, [5001]);
});

test("probes health from the Host and rejects a non-http endpoint", async (t) => {
  const { manager } = managerIn(t);
  const health = await manager.health("http://127.0.0.1:2334");
  assert.equal(health.endpoint, "http://127.0.0.1:2334");
  assert.equal(health.ok, false);
  assert.equal(typeof health.checkedAt, "string");
  await assert.rejects(() => manager.health("file:///etc/passwd"), /http/);
});

test("shows the rotated service log when the current one is empty", (t) => {
  const { manager, directory } = managerIn(t);
  writeFileSync(join(directory, "service.log"), "");
  writeFileSync(join(directory, "service.log.1"), "Traceback: boom\n");
  const log = manager.serviceLog(directory);
  // The path names the file actually read, so the UI cannot label a rotated
  // backup as the live log.
  assert.equal(log.logPath, join(directory, "service.log.1"));
  assert.match(log.lines, /boom/);
});

test("validates the merged .env, not just the submitted values", async (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(
    envPath,
    "LLM_MODEL_NAME=qwen3.7-plus\nLLM_API_KEY=sk-keep\nEMBEDDING_API_KEY=EMPTY\n",
  );

  // An untouched required key still satisfies the requirement.
  const summary = manager.writeEnv(directory, {
    LLM_MODEL_NAME: "qwen3.8-plus",
  });
  assert.equal(
    summary.entries.find((entry) => entry.key === "LLM_MODEL_NAME").value,
    "qwen3.8-plus",
  );
  assert.match(readFileSync(envPath, "utf8"), /^LLM_API_KEY=sk-keep$/m);

  // An emptied secret never reaches the file: "leave blank to keep" wins over
  // an explicit clear, so the stored key survives and the file is unchanged.
  manager.writeEnv(directory, { LLM_API_KEY: "" });
  assert.match(readFileSync(envPath, "utf8"), /^LLM_API_KEY=sk-keep$/m);
});

test("writes .env atomically with a backup and owner-only mode", (t) => {
  const { manager, directory } = managerIn(t);
  const envPath = join(directory, ".env");
  writeFileSync(
    envPath,
    "LLM_API_KEY=sk-old\nEMBEDDING_API_KEY=EMPTY\nLLM_MODEL_NAME=old\n",
  );

  manager.writeEnv(directory, { LLM_MODEL_NAME: "new" });
  assert.match(readFileSync(envPath, "utf8"), /^LLM_MODEL_NAME=new$/m);
  const backup = readFileSync(`${envPath}.bak`, "utf8");
  assert.match(backup, /^LLM_MODEL_NAME=old$/m);
  assert.match(backup, /^LLM_API_KEY=sk-old$/m);
  if (process.platform !== "win32")
    assert.equal(statSync(envPath).mode & 0o777, 0o600);
});
