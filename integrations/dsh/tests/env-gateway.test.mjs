import assert from "node:assert/strict";
import test from "node:test";
import { Context } from "@deepseek-ai/cordis";

import { ReMeEnvGateway } from "../dist/environment/host/gateway.js";

/** Manager stub: no subprocesses, just enough surface for the gateway. */
function stubManager() {
  const calls = [];
  return {
    calls,
    detectPython: async () => ({
      platform: "win32",
      scannedAt: "",
      envs: [],
      suggested: "",
      condaRoot: "",
      errors: [],
    }),
    checkReme: async (pythonPath) => ({
      installed: false,
      version: "",
      versionOk: false,
      hasCore: false,
      cliPath: "",
      error: pythonPath ? "" : "Python interpreter path must be absolute",
    }),
    health: async (endpoint, authToken) => ({
      endpoint,
      ok: false,
      version: "",
      error: authToken ? "auth token forwarded" : "",
      checkedAt: "",
    }),
    installStatus: async (jobId, pythonPath) => {
      calls.push({ method: "installStatus", jobId, pythonPath });
      return null;
    },
    serviceForget: async (endpoint, expectPid) => {
      calls.push({ method: "serviceForget", endpoint, expectPid });
      return {
        forgotten: false,
        reason: endpoint,
        code: "no-record",
      };
    },
  };
}

/** Mount the gateway inside the same isolated realm the patch creates. */
async function mounted() {
  const root = new Context();
  let realm;
  let manager;
  await root.plugin((ctx) => {
    realm = ctx.isolate({ remeMemory: true });
  });
  await realm.plugin((ctx) => {
    ctx.provide("remeMemory", {});
    manager = stubManager();
    ctx.provide("remeEnv", manager);
  });
  await realm.plugin(ReMeEnvGateway);
  return { root, realm, manager };
}

test("gateway mounts without a service-key conflict and keeps the wire namespace", async () => {
  const { root } = await mounted();
  const receiver = root.reflect.get("remeEnvGateway", false);
  assert.ok(receiver, "gateway service is registered under its own key");
  assert.ok(receiver.typertRemote, "gateway exposes a typertRemote binding");
  assert.equal(receiver.typertRemote.serviceKey, "remeEnvGateway");
  assert.equal(receiver.typertRemote.namespace, "remeEnv");
  // The manager must stay reachable under the key the methods inject.
  assert.ok(root.reflect.get("remeEnv", false));
});

test("SRC discovery exposes remeEnv/python with one plain input parameter", async () => {
  const { root } = await mounted();
  let binding;
  for (const [serviceKey, definition] of Object.entries(root.reflect.props)) {
    if (definition.type !== "service") continue;
    const receiver = root.reflect.get(serviceKey, false);
    if (!receiver?.typertRemote) continue;
    if (receiver.typertRemote.namespace !== "remeEnv") continue;
    binding = { serviceKey, receiver };
  }
  assert.ok(binding, "an active service exports the remeEnv namespace");
  assert.equal(binding.serviceKey, "remeEnvGateway");

  const source = Function.prototype.toString.call(
    Object.getPrototypeOf(binding.receiver).python,
  );
  const open = source.indexOf("(");
  const close = source.indexOf(")", open + 1);
  const parameters = source
    .slice(open + 1, close)
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  // The Gateway parses this source, so defaults, destructuring, and rest
  // parameters would make the endpoint call fail at runtime.
  assert.deepEqual(parameters, ["input"]);

  const result = await binding.receiver.python({ refresh: true });
  assert.equal(result.platform, "win32");
});

test("gateway methods remain callable through the manager after remount", async () => {
  const { root } = await mounted();
  const receiver = root.reflect.get("remeEnvGateway", false);
  const check = await receiver.checkReme({ pythonPath: "" });
  assert.equal(check.installed, false);
  assert.match(check.error, /absolute/);
});

test("health keeps the single-input convention and forwards the token", async () => {
  const { root } = await mounted();
  const receiver = root.reflect.get("remeEnvGateway", false);
  const source = Function.prototype.toString.call(
    Object.getPrototypeOf(receiver).health,
  );
  const open = source.indexOf("(");
  const close = source.indexOf(")", open + 1);
  const parameters = source
    .slice(open + 1, close)
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  assert.deepEqual(parameters, ["input"]);

  const result = await receiver.health({
    endpoint: "http://127.0.0.1:2334",
    authToken: "secret",
  });
  assert.equal(result.endpoint, "http://127.0.0.1:2334");
  assert.equal(result.error, "auth token forwarded");
});

test("serviceForget keeps the single-input convention and forwards the endpoint", async () => {
  const { root } = await mounted();
  const receiver = root.reflect.get("remeEnvGateway", false);
  const source = Function.prototype.toString.call(
    Object.getPrototypeOf(receiver).serviceForget,
  );
  const open = source.indexOf("(");
  const close = source.indexOf(")", open + 1);
  const parameters = source
    .slice(open + 1, close)
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  assert.deepEqual(parameters, ["input"]);

  const result = await receiver.serviceForget({
    endpoint: "http://127.0.0.1:2334",
  });
  assert.equal(result.forgotten, false);
  assert.equal(result.reason, "http://127.0.0.1:2334");
});

test("the gateway forwards the interpreter lookup and the pinned record", async () => {
  const { root, manager } = await mounted();
  const receiver = root.reflect.get("remeEnvGateway", false);

  // The tab restores the running job of one interpreter, not the newest job.
  await receiver.installStatus({ pythonPath: "C:\\env\\a\\python.exe" });
  const status = manager.calls.find((call) => call.method === "installStatus");
  assert.equal(status.jobId, "");
  assert.equal(status.pythonPath, "C:\\env\\a\\python.exe");

  // The confirmed recovery targets the record the Host named, pinned by pid.
  await receiver.serviceForget({
    endpoint: "http://127.0.0.1:2333",
    expectPid: 6000,
  });
  const forget = manager.calls.find((call) => call.method === "serviceForget");
  assert.equal(forget.endpoint, "http://127.0.0.1:2333");
  assert.equal(forget.expectPid, 6000);
});
