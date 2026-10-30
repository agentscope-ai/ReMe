import assert from "node:assert/strict";
import test from "node:test";

import React from "react";
import { act, create } from "react-test-renderer";

import { ReMeEnvironmentTab, envZh } from "../dist/environment/client/tab.js";
import { fill } from "../dist/environment/client/locales.js";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const CONDA = "D:\\ProgramFils\\Miniconda";
const SUGGESTED = `${CONDA}\\envs\\reme-serve\\python.exe`;

function env(name, version, reme) {
  const root = name === "base" ? CONDA : `${CONDA}\\envs\\${name}`;
  return {
    exe: `${root}\\python.exe`,
    version,
    ok: true,
    origin: { kind: "conda", envName: name, root },
    source: "conda env list",
    ...(reme ? { reme: { installed: true, version: reme } } : {}),
  };
}

const ENVS = [
  env("base", "3.12.3", "0.4.0.9"),
  env("com_env", "3.12.9"),
  env("copaw", "3.12.0", "0.3.1.8"),
  env("drama", "3.12.13"),
  env("mochi", "3.12.13"),
  env("mochi_test", "3.12.13"),
  env("nl_agent", "3.12.0"),
  env("reme-serve", "3.12.13", "0.4.0.9"),
];

const RESPONSES = {
  "remeEnv/python": {
    platform: "win32",
    scannedAt: new Date().toISOString(),
    envs: ENVS,
    suggested: SUGGESTED,
    condaRoot: CONDA,
    errors: [],
  },
  "remeEnv/checkReme": {
    installed: true,
    version: "0.4.0.9",
    versionOk: true,
    hasCore: true,
    cliPath: `${CONDA}\\envs\\reme-serve\\Scripts\\reme.exe`,
    error: "",
  },
  "remeEnv/installStatus": null,
  "remeEnv/readEnv": {
    workspaceDir: "C:\\Users\\test\\.reme",
    exists: true,
    envPath: "C:\\Users\\test\\.reme\\.env",
    entries: [
      {
        key: "LLM_BACKEND",
        value: "dashscope",
        masked: "dashscope",
        secret: false,
        set: true,
      },
      {
        key: "LLM_MODEL_NAME",
        value: "qwen3.7-plus",
        masked: "qwen3.7-plus",
        secret: false,
        set: true,
      },
    ],
  },
  "remeEnv/serviceStatus": {
    phase: "running",
    version: "0.4.0.9",
    managed: false,
    pid: 98820,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  },
};

const t = (key) => envZh[key];

function textOf(value) {
  if (Array.isArray(value)) return value.map(textOf).join("");
  return typeof value === "string" ? value : "";
}

function classesOf(node, found = new Map()) {
  if (!node || typeof node === "string") return found;
  if (Array.isArray(node)) {
    node.forEach((child) => classesOf(child, found));
    return found;
  }
  for (const name of String(node.props.className ?? "").split(" ")) {
    if (name) found.set(name, (found.get(name) ?? 0) + 1);
  }
  (node.children ?? []).forEach((child) => classesOf(child, found));
  return found;
}

function texts(node, found = []) {
  if (!node) return found;
  if (typeof node === "string") {
    found.push(node);
    return found;
  }
  if (Array.isArray(node)) {
    node.forEach((child) => texts(child, found));
    return found;
  }
  (node.children ?? []).forEach((child) => texts(child, found));
  return found;
}

async function mount(overrides = {}, options = {}) {
  const calls = { mutate: [], rpc: [] };
  const responses = { ...RESPONSES, ...(options.responses ?? {}) };
  const listeners = new Set();
  // A new object per change, the way the Host-backed form publishes a snapshot.
  let snapshot = {
    status: "ready",
    writable: true,
    revision: 3,
    value: {
      endpoint: "http://127.0.0.1:2333",
      pythonPath: "",
      remeWorkspaceDir: "",
      autoStart: false,
      requestTimeoutMs: 10000,
      backgroundTimeoutMs: 3600000,
      ...overrides,
    },
  };
  const scope = {
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    getSnapshot: () => snapshot,
    mutate: async (ops, revision) => {
      calls.mutate.push({ ops, revision });
      return options.mutate ? options.mutate() : true;
    },
  };
  const rpc = {
    call: async (_path, method, payload, signal) => {
      calls.rpc.push({ method, input: payload?.args?.input });
      if (options.pending?.includes(method))
        return new Promise((_resolve, reject) => {
          signal?.addEventListener(
            "abort",
            () => reject(new Error("aborted")),
            { once: true },
          );
        });
      const value = responses[method];
      return {
        ok: true,
        value:
          typeof value === "function"
            ? value(payload?.args?.input, signal)
            : value ?? null,
      };
    },
  };

  let renderer;
  await act(async () => {
    renderer = create(
      React.createElement(ReMeEnvironmentTab, { scope, rpc, t }),
    );
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
  return {
    renderer,
    calls,
    setValue(patch) {
      snapshot = {
        ...snapshot,
        value: { ...snapshot.value, ...patch },
      };
      for (const listener of listeners) listener();
    },
    restore: () => {
      // Unmount so the effect cleanup paths (aborts, timers) actually run.
      act(() => {
        renderer.unmount();
      });
      listeners.clear();
    },
  };
}

function buttonByText(renderer, label) {
  const found = renderer.root.findAll(
    (node) => node.type === "button" && textOf(node.props.children) === label,
  );
  assert.ok(found.length > 0, `button not found: ${label}`);
  return found[0];
}

test("renders the status hero and the four setup steps", async (t) => {
  const { renderer, restore } = await mount();
  t.after(restore);
  const json = renderer.toJSON();
  const classes = classesOf(json);
  const copy = texts(json).join(" ");

  assert.equal(classes.get("reme-env-hero"), 1);
  assert.equal(classes.get("reme-env-card"), 4);
  assert.equal(classes.get("reme-env-picked"), 1);
  assert.equal(classes.get("reme-env-metrics"), 1);
  assert.equal(classes.get("reme-env-autostart"), 1);

  assert.match(copy, /ReMe 本地运行时/);
  assert.match(copy, /已就绪，记忆保存在本地/);
  assert.match(copy, /http:\/\/127\.0\.0\.1:2333/);
  assert.match(copy, /8 个环境/);
  assert.match(copy, /服务 · ReMe 0\.4\.0\.9/);
  assert.match(copy, /Python 3\.12\.13/);
  assert.match(copy, /reme-ai 0\.4\.0\.9/);
  assert.match(copy, /C:\\Users\\test\\\.reme\\\.env/);
});

test("keeps the interpreter list collapsed until asked, then paginates", async (t) => {
  const { renderer, restore } = await mount();
  t.after(restore);
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), undefined);

  await act(async () => {
    buttonByText(renderer, envZh.changeInterpreter).props.onClick();
  });
  const expanded = classesOf(renderer.toJSON());
  assert.equal(expanded.get("reme-env-option"), 6);
  assert.equal(expanded.get("reme-env-search"), 1);
  assert.equal(expanded.get("reme-env-more"), 1);

  await act(async () => {
    renderer.root
      .findAll(
        (node) =>
          node.type === "button" && textOf(node.props.children).includes("8"),
      )[0]
      .props.onClick();
  });
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), 8);
});

test("marks the suggested interpreter and explains why", async (t) => {
  const { renderer, calls, restore } = await mount();
  t.after(restore);
  const json = renderer.toJSON();
  const recommended = renderer.root.findAll(
    (node) =>
      String(node.props.className ?? "").includes("reme-env-tag brand") &&
      textOf(node.props.children) === envZh.recommended,
  );
  assert.ok(recommended.length >= 1);
  assert.equal(recommended[0].props.title, envZh.recommendWhyReme);
  assert.ok(texts(json).join(" ").includes(SUGGESTED));

  await act(async () => {
    buttonByText(renderer, envZh.changeInterpreter).props.onClick();
  });
  const radios = renderer.root.findAll(
    (node) => node.type === "input" && node.props.type === "radio",
  );
  // The current interpreter stays in the group, checked and disabled, so
  // assistive technology can announce the selection.
  assert.equal(radios.filter((node) => node.props.checked === true).length, 1);
  assert.equal(
    radios.find((node) => node.props.checked === true).props.disabled,
    true,
  );
  const target = radios.find((node) => node.props.disabled === false);
  assert.ok(target);
  await act(async () => {
    target.props.onChange();
  });
  assert.deepEqual(calls.mutate.at(-1)?.ops, [
    { op: "set", path: ["pythonPath"], value: ENVS[0].exe },
  ]);
});

test("sends the chosen package mirror with the install request", async (t) => {
  const { renderer, calls, restore } = await mount();
  t.after(restore);

  const select = renderer.root.findAll((node) => node.type === "select")[0];
  await act(async () => {
    select.props.onChange({
      target: { value: "https://pypi.tuna.tsinghua.edu.cn/simple" },
    });
  });
  await act(async () => {
    buttonByText(renderer, envZh.reinstall).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.confirmInstall).props.onClick();
  });

  const install = calls.rpc.filter((call) => call.method === "remeEnv/install");
  assert.ok(install.length > 0, "the install request must be sent");
  assert.deepEqual(install.at(-1).input, {
    pythonPath: SUGGESTED,
    upgrade: false,
    indexUrl: "https://pypi.tuna.tsinghua.edu.cn/simple",
    forceReinstall: true,
  });
});

test("drops .env drafts when the workspace changes", async (t) => {
  const { renderer, calls, setValue, restore } = await mount();
  t.after(restore);

  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  const apiKey = () =>
    renderer.root.findAll(
      (node) => node.type === "input" && node.props.type === "password",
    )[0];
  await act(async () => {
    apiKey().props.onChange({
      target: { value: "sk-typed-for-first-workspace" },
    });
  });
  assert.equal(apiKey().props.value, "sk-typed-for-first-workspace");

  await act(async () => {
    setValue({ remeWorkspaceDir: "/tmp/second-workspace" });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  // The draft belonged to the first workspace and is gone with it, so a save
  // cannot carry a key typed for one workspace into another.
  assert.equal(apiKey().props.value, "");

  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  const write = calls.rpc
    .filter((call) => call.method === "remeEnv/writeEnv")
    .at(-1);
  assert.ok(write);
  assert.equal(write.input.workspaceDir, "/tmp/second-workspace");
  assert.deepEqual(write.input.values, {});
});

test("probes the service from the Host, never from the browser", async (t) => {
  const original = globalThis.fetch;
  let fetched = 0;
  globalThis.fetch = async () => {
    fetched += 1;
    throw new Error("the tab must not fetch the service URL itself");
  };
  t.after(() => {
    globalThis.fetch = original;
  });
  const { renderer, calls, restore } = await mount();
  t.after(restore);
  assert.equal(fetched, 0);
  assert.match(
    texts(renderer.toJSON()).join(" "),
    /服务 · ReMe 0\.4\.0\.9/,
    "the hero reports the version the Host service probe returned",
  );
  assert.ok(
    calls.rpc.some((call) => call.method === "remeEnv/serviceStatus"),
    "the hero reads its connection from the Host service probe",
  );
  assert.equal(
    calls.rpc.some((call) => call.method === "remeEnv/health"),
    false,
    "a local endpoint is covered by the service probe already",
  );
});

test("a remote endpoint never runs the local setup steps", async (t) => {
  const { calls, restore } = await mount(
    { endpoint: "http://10.0.0.5:2333" },
    {
      responses: {
        "remeEnv/health": {
          endpoint: "http://10.0.0.5:2333",
          ok: false,
          version: "",
          error: "unreachable",
          checkedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  const methods = calls.rpc.map((call) => call.method);
  assert.ok(
    methods.includes("remeEnv/health"),
    "a remote endpoint is only health-checked",
  );
  for (const local of [
    "remeEnv/python",
    "remeEnv/checkReme",
    "remeEnv/readEnv",
    "remeEnv/serviceStatus",
  ])
    assert.equal(methods.includes(local), false, `${local} must not run`);
});

test("shows why Stop left the service alone", async (t) => {
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": {
          phase: "running",
          managed: true,
          pid: 98820,
          version: "0.4.0.9",
          endpoint: "http://127.0.0.1:2333",
          logPath: "C:\\Users\\test\\.reme\\service.log",
        },
        "remeEnv/serviceStop": {
          stopped: false,
          reason: "The recorded process identity could not be verified",
          code: "unverified",
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  // The Host code is localized in the UI instead of leaking its English text.
  assert.ok(
    texts(renderer.toJSON()).join(" ").includes(envZh.stopUnknownIdentity),
  );
  // Stop can never succeed for this record, so the record-only way out is
  // offered next to the explanation.
  assert.ok(buttonByText(renderer, envZh.forgetRecord));
});

test("a process that survived the force-kill still offers forgetting the record", async (t) => {
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": {
          phase: "running",
          managed: true,
          pid: 98820,
          version: "0.4.0.9",
          endpoint: "http://127.0.0.1:2333",
          logPath: "C:\\Users\\test\\.reme\\service.log",
        },
        "remeEnv/serviceStop": {
          stopped: false,
          reason: "The service did not exit after the force-kill",
          code: "not-exited",
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.stopNotExited));
  // Start would refuse with `previous-not-stopped`, so the record must be
  // forgettable instead of leaving no exit at all.
  assert.ok(buttonByText(renderer, envZh.forgetRecord));
});

test("surfaces a rejected settings write", async (t) => {
  const { renderer, restore } = await mount({}, { mutate: async () => false });
  t.after(restore);
  const checkbox = renderer.root.findAll(
    (node) => node.type === "input" && node.props.type === "checkbox",
  )[0];
  assert.ok(checkbox);
  await act(async () => {
    checkbox.props.onChange({ target: { checked: true } });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  assert.match(texts(renderer.toJSON()).join(" "), /设置未保存/);
});

test("an aborted LLM test releases its button", async (t) => {
  const { renderer, setValue, restore } = await mount(
    {},
    { pending: ["remeEnv/testLlm"] },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.testLlmButton).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  assert.ok(texts(renderer.toJSON()).join(" ").includes(envZh.testingLlm));

  await act(async () => {
    setValue({ remeWorkspaceDir: "/tmp/second-workspace" });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.equal(copy.includes(envZh.testingLlm), false);
  assert.ok(copy.includes(envZh.testLlmButton), "the button is usable again");
});

test("a start that died before listening reports immediately", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": stopped,
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_200));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.startFailedStopped));
  // The message is already a complete sentence; prefixing it would read
  // "Service operation failed: The service exited…".
  assert.equal(copy.includes(`${envZh.serviceError}:`), false);
  // Ready to retry without waiting out the 120 s deadline.
  assert.ok(buttonByText(renderer, envZh.startService));
});

test("a start that lands on a foreign conflict shows the conflict state", async (t) => {
  let calls = 0;
  const states = [
    {
      phase: "stopped",
      managed: false,
      endpoint: "http://127.0.0.1:2333",
      logPath: "C:\\Users\\test\\.reme\\service.log",
    },
    {
      phase: "conflict",
      managed: false,
      endpoint: "http://127.0.0.1:2333",
      logPath: "C:\\Users\\test\\.reme\\service.log",
    },
  ];
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": () =>
          states[Math.min(calls++, states.length - 1)],
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: states[0].logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_200));
  });
  assert.ok(texts(renderer.toJSON()).join(" ").includes(envZh.conflictNote));
});

test("Reinstall forces a reinstall while install and upgrade use the right flags", async () => {
  const upgrade = await mount(
    {},
    {
      responses: {
        "remeEnv/checkReme": {
          installed: true,
          version: "0.3.1.8",
          versionOk: false,
          hasCore: false,
          cliPath: "",
          error: "",
        },
      },
    },
  );
  await act(async () => {
    buttonByText(upgrade.renderer, envZh.upgradeButton).props.onClick();
  });
  await act(async () => {
    buttonByText(upgrade.renderer, envZh.confirmInstall).props.onClick();
  });
  const upgradeCall = upgrade.calls.rpc
    .filter((call) => call.method === "remeEnv/install")
    .at(-1);
  assert.deepEqual(upgradeCall.input, {
    pythonPath: SUGGESTED,
    upgrade: true,
  });
  upgrade.restore();

  const fresh = await mount(
    {},
    {
      responses: {
        "remeEnv/checkReme": {
          installed: false,
          version: "",
          versionOk: false,
          hasCore: false,
          cliPath: "",
          error: "",
        },
      },
    },
  );
  await act(async () => {
    buttonByText(fresh.renderer, envZh.installButton).props.onClick();
  });
  await act(async () => {
    buttonByText(fresh.renderer, envZh.confirmInstall).props.onClick();
  });
  const freshCall = fresh.calls.rpc
    .filter((call) => call.method === "remeEnv/install")
    .at(-1);
  assert.deepEqual(freshCall.input, {
    pythonPath: SUGGESTED,
    upgrade: false,
  });
  fresh.restore();

  const reinstall = await mount();
  await act(async () => {
    buttonByText(reinstall.renderer, envZh.reinstall).props.onClick();
  });
  // The confirmation copy follows the Reinstall CTA, not the first-install one.
  assert.ok(
    texts(reinstall.renderer.toJSON())
      .join(" ")
      .includes(envZh.reinstallConfirm.split("{path}")[0]),
  );
  await act(async () => {
    buttonByText(reinstall.renderer, envZh.confirmInstall).props.onClick();
  });
  const reinstallCall = reinstall.calls.rpc
    .filter((call) => call.method === "remeEnv/install")
    .at(-1);
  assert.deepEqual(reinstallCall.input, {
    pythonPath: SUGGESTED,
    upgrade: false,
    forceReinstall: true,
  });
  reinstall.restore();
});

test("localizes a coded start error instead of showing Host English", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": stopped,
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "Another process is already listening on this endpoint",
          code: "conflict",
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 10));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.startErrorConflict));
  assert.equal(copy.includes("Another process is already listening"), false);
});

test("switching the endpoint aborts the previous start poll", async (t) => {
  const OLD = "http://127.0.0.1:2333";
  const NEW = "http://127.0.0.1:2444";
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: OLD,
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const newRunning = {
    phase: "running",
    managed: true,
    pid: 1111,
    endpoint: NEW,
    version: "9.9.9",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const oldRunning = {
    ...newRunning,
    pid: 5000,
    endpoint: OLD,
    version: "0.4.0.9",
  };
  let statusCalls = 0;
  const { renderer, setValue, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": (input) => {
          if (input?.endpoint === NEW) return newRunning;
          statusCalls += 1;
          return statusCalls === 1 ? stopped : oldRunning;
        },
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  // Switch before the poll's 1 s tick: it must not paint the old endpoint's
  // running data under the new URL.
  await act(async () => {
    setValue({ endpoint: NEW });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_200));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("PID 1111"), "the new endpoint's pid is shown");
  assert.equal(copy.includes("PID 5000"), false);
});

test("a live record that never listened offers Stop instead of 'Starting'", async (t) => {
  // The Host reports a recorded launch that is still alive but not listening
  // as `stopped` + pid. Mapping that to `starting` on every read used to lock
  // the card: Start disabled, no Stop, and Recheck repeated the same answer.
  const stale = {
    phase: "stopped",
    managed: false,
    pid: 6000,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, restore } = await mount(
    {},
    { responses: { "remeEnv/serviceStatus": stale } },
  );
  t.after(restore);
  const copy = texts(renderer.toJSON()).join(" ");
  assert.equal(copy.includes(envZh.serviceBadgeStarting), false);
  assert.ok(copy.includes(envZh.serviceBadgeStale));
  assert.ok(copy.includes(envZh.staleNote));
  // Stop is the way out for a provably owned leftover...
  const stop = buttonByText(renderer, envZh.stopService);
  assert.equal(stop.props.disabled, false, "the recorded pid can be stopped");
  // ...and Start must stay available too: the pid may have been recycled by a
  // foreign process, where Stop refuses and only a restart self-heals.
  const start = buttonByText(renderer, envZh.startService);
  assert.equal(start.props.disabled, false, "Start can replace the record");
});

test("Start replaces a stale record that belongs to a foreign pid", async (t) => {
  // The pid is alive but no longer our service, so Stop answers `foreign` and
  // keeps the record. Start must still be offered: the Host discards a foreign
  // record and launches a fresh service.
  const stale = {
    phase: "stopped",
    managed: false,
    pid: 6000,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const running = {
    phase: "running",
    managed: true,
    pid: 7000,
    version: "0.4.0.9",
    endpoint: "http://127.0.0.1:2333",
    logPath: stale.logPath,
  };
  let statusCalls = 0;
  const { renderer, calls, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": () => {
          statusCalls += 1;
          return statusCalls === 1 ? stale : running;
        },
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: stale.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_200));
  });
  assert.ok(
    calls.rpc.some((call) => call.method === "remeEnv/serviceStart"),
    "the restart really reached the Host",
  );
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("ReMe 0.4.0.9"));
  assert.ok(copy.includes("PID 7000"));
});

test("an owned conflict during a start keeps polling until healthy", async (t) => {
  const stop = (pid) => ({
    phase: "stopped",
    managed: false,
    ...(pid === undefined ? {} : { pid }),
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  });
  const conflict = {
    phase: "conflict",
    managed: true,
    pid: 6000,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const running = {
    phase: "running",
    managed: true,
    pid: 6000,
    version: "0.4.0.9",
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  let statusCalls = 0;
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": () => {
          statusCalls += 1;
          if (statusCalls === 1) return stop();
          if (statusCalls === 2) return conflict;
          return running;
        },
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: running.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_200));
  });
  // One owned conflict is our own launch booting; it must not become a Stop
  // offer that would kill a service which is about to answer.
  assert.ok(
    texts(renderer.toJSON()).join(" ").includes(envZh.serviceBadgeStarting),
  );
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 1_100));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("ReMe 0.4.0.9"));
  assert.equal(copy.includes(envZh.conflictManagedNote), false);
});

test("a Host conflict refusal shows the conflict card, not an error", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const conflict = {
    phase: "conflict",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: stopped.logPath,
  };
  let statusCalls = 0;
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": () => {
          statusCalls += 1;
          return statusCalls === 1 ? stopped : conflict;
        },
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "Another process is already listening on this endpoint",
          code: "conflict",
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.conflictNote));
  assert.equal(copy.includes(envZh.startErrorConflict), false);
});

test("a slow status answer does not overwrite a finished start", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const running = {
    phase: "running",
    managed: true,
    pid: 7000,
    version: "0.4.0.9",
    endpoint: "http://127.0.0.1:2333",
    logPath: stopped.logPath,
  };
  let statusCalls = 0;
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": () => {
          statusCalls += 1;
          if (statusCalls === 1) return stopped;
          // The manual Recheck answers late, after the start poll has settled.
          if (statusCalls === 2)
            return new Promise((resolve) =>
              setTimeout(() => resolve(stopped), 1_500),
            );
          return running;
        },
        "remeEnv/serviceStart": {
          started: true,
          alreadyRunning: false,
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  const rechecks = renderer.root.findAll(
    (node) =>
      node.type === "button" && textOf(node.props.children) === envZh.recheck,
  );
  await act(async () => {
    rechecks.at(-1).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 2_300));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("ReMe 0.4.0.9"));
  assert.equal(copy.includes(envZh.serviceBadgeStopped), false);
});

test("a Stop result for the previous endpoint is dropped", async (t) => {
  const OLD = "http://127.0.0.1:2333";
  const NEW = "http://127.0.0.1:2444";
  const oldRunning = {
    phase: "running",
    managed: true,
    pid: 5000,
    version: "0.4.0.9",
    endpoint: OLD,
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const newRunning = {
    ...oldRunning,
    pid: 1111,
    version: "9.9.9",
    endpoint: NEW,
  };
  let release;
  const stopResult = new Promise((resolve) => {
    release = resolve;
  });
  const { renderer, setValue, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": (input) =>
          input?.endpoint === NEW ? newRunning : oldRunning,
        "remeEnv/serviceStop": () => stopResult,
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.stopService).props.onClick();
  });
  await act(async () => {
    setValue({ endpoint: NEW });
  });
  await act(async () => {
    release({
      stopped: false,
      reason: "could not be verified",
      code: "unverified",
    });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.equal(copy.includes(envZh.stopUnknownIdentity), false);
  const stops = renderer.root.findAll(
    (node) =>
      node.type === "button" &&
      textOf(node.props.children) === envZh.stopService,
  );
  assert.ok(stops.length > 0);
  assert.equal(stops[0].props.disabled, false);
});

test("localizes a Host health failure code", async (t) => {
  const { renderer, restore } = await mount(
    { endpoint: "http://10.0.0.5:2333" },
    {
      responses: {
        "remeEnv/health": {
          endpoint: "http://10.0.0.5:2333",
          ok: false,
          version: "",
          error: "The endpoint did not answer with a ReMe health report",
          code: "not-reme",
          checkedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.healthErrorNotReme));
  assert.equal(
    copy.includes("did not answer with a ReMe health report"),
    false,
  );
});

test("a Python refresh keeps the previous interpreter list on screen", async (t) => {
  let pythonCalls = 0;
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/python": () => {
          pythonCalls += 1;
          // The refresh never answers; the previous detection must stay.
          return pythonCalls === 1
            ? RESPONSES["remeEnv/python"]
            : new Promise(() => {});
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.changeInterpreter).props.onClick();
  });
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), 6);
  await act(async () => {
    buttonByText(renderer, envZh.refresh).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.equal(copy.includes(envZh.emptyTitle), false);
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), 6);
});

test("a failed Python refresh keeps the working list without the empty state", async (t) => {
  let pythonCalls = 0;
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/python": () => {
          pythonCalls += 1;
          if (pythonCalls === 1) return RESPONSES["remeEnv/python"];
          throw new Error("probe exploded");
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.changeInterpreter).props.onClick();
  });
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), 6);
  await act(async () => {
    buttonByText(renderer, envZh.refresh).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.detectFailed));
  // A failed probe that still has the previous result must not claim that no
  // Python exists, nor offer Miniconda next to a valid interpreter list.
  assert.equal(copy.includes(envZh.emptyTitle), false);
  assert.equal(classesOf(renderer.toJSON()).get("reme-env-option"), 6);
});

test("the conflict fallback of a start does not paint an aborted request", async (t) => {
  const OLD = "http://127.0.0.1:2333";
  const NEW = "http://127.0.0.1:2444";
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: OLD,
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const newRunning = {
    phase: "running",
    managed: true,
    pid: 1111,
    version: "9.9.9",
    endpoint: NEW,
    logPath: stopped.logPath,
  };
  let statusCalls = 0;
  const { renderer, setValue, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": (input, signal) => {
          if (input?.endpoint === NEW) return newRunning;
          statusCalls += 1;
          if (statusCalls === 1) return stopped;
          // The R3 re-query hangs until the endpoint switch aborts it.
          return new Promise((_resolve, reject) => {
            signal?.addEventListener(
              "abort",
              () => reject(new Error("aborted")),
              { once: true },
            );
          });
        },
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "Another process is already listening on this endpoint",
          code: "conflict",
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 20));
  });
  await act(async () => {
    setValue({ endpoint: NEW });
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("ReMe 9.9.9"), "the new endpoint state is shown");
  // The old endpoint's conflict error must not land on the new card.
  assert.equal(copy.includes(envZh.startErrorConflict), false);
});

test("an unverified record reports the pid it refused to touch", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, calls, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": stopped,
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "The recorded service identity could not be verified",
          code: "unverified-record",
          pid: 6000,
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
        "remeEnv/serviceForget": { forgotten: true, pid: 6000 },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes("PID 6000"));
  assert.equal(copy.includes("The recorded service identity"), false);

  // Both Start and Stop are guaranteed to fail here, so the record-only way
  // out must be offered instead of leaving the card with no working action.
  assert.ok(copy.includes(envZh.forgetHint));
  assert.ok(buttonByText(renderer, envZh.forgetRecord));
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  assert.ok(
    texts(renderer.toJSON())
      .join(" ")
      .includes(
        fill(envZh.forgetConfirmRecord, { endpoint: "http://127.0.0.1:2333" }),
      ),
    "the confirmation names the record it forgets",
  );
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  assert.ok(
    calls.rpc.some((call) => call.method === "remeEnv/serviceForget"),
    "forgetting reaches the Host",
  );
  const after = texts(renderer.toJSON()).join(" ");
  assert.equal(after.includes(envZh.forgetHint), false);
  assert.equal(after.includes(envZh.forgetRecord), false);
});

test("a successful install refreshes the dependency check", async (t) => {
  // The card used to keep reporting "not installed" after a successful
  // install, so Start stayed disabled until the user pressed Check by hand.
  let checkCalls = 0;
  const job = {
    id: "install-1",
    phase: "running",
    pythonPath: SUGGESTED,
    command: "pip install reme-ai[core]",
    startedAt: new Date().toISOString(),
    logPath: "C:\\Users\\test\\.reme\\install-1.log",
    logTail: "",
  };
  const { renderer, calls, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/checkReme": () => {
          checkCalls += 1;
          return checkCalls === 1
            ? {
                installed: false,
                version: "",
                versionOk: false,
                hasCore: false,
                cliPath: "",
                error: "",
              }
            : {
                installed: true,
                version: "0.4.0.9",
                versionOk: true,
                hasCore: true,
                cliPath: "",
                error: "",
              };
        },
        "remeEnv/install": () => job,
        "remeEnv/installStatus": (input) =>
          input?.jobId
            ? {
                ...job,
                phase: "succeeded",
                finishedAt: new Date().toISOString(),
              }
            : null,
        "remeEnv/serviceStatus": {
          phase: "stopped",
          managed: false,
          endpoint: "http://127.0.0.1:2333",
          logPath: "C:\\Users\\test\\.reme\\service.log",
        },
      },
    },
  );
  t.after(restore);
  assert.ok(texts(renderer.toJSON()).join(" ").includes(envZh.notInstalled));

  await act(async () => {
    buttonByText(renderer, envZh.installButton).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.confirmInstall).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });

  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(
    copy.includes(envZh.installSucceeded),
    "the install reports success",
  );
  // The dependency check ran again for this interpreter.
  const checks = calls.rpc.filter(
    (call) => call.method === "remeEnv/checkReme",
  );
  assert.ok(checks.length >= 2, "the check is rerun after the install");
  assert.ok(
    copy.includes("reme-ai 0.4.0.9"),
    "the card shows the freshly installed version",
  );
  // Start is usable again now that the dependency is there.
  assert.equal(
    buttonByText(renderer, envZh.startService).props.disabled,
    false,
  );
});

test("an install answer for another interpreter is dropped", async (t) => {
  let release;
  const pending = new Promise((resolve) => {
    release = resolve;
  });
  const job = {
    id: "install-9",
    phase: "running",
    pythonPath: SUGGESTED,
    command: "pip install reme-ai[core]",
    startedAt: new Date().toISOString(),
    logPath: "C:\\Users\\test\\.reme\\install-9.log",
    logTail: "",
  };
  const { renderer, setValue, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/install": () => pending,
        // The Host keeps reporting the job as running, so a job that leaks
        // into the new interpreter stays visible instead of being wiped by the
        // next poll.
        "remeEnv/installStatus": (input) => (input?.jobId ? job : null),
      },
    },
  );
  t.after(restore);

  await act(async () => {
    buttonByText(renderer, envZh.reinstall).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.confirmInstall).props.onClick();
  });
  // The interpreter changes while the install request is still in flight.
  await act(async () => {
    setValue({ pythonPath: ENVS[0].exe });
  });
  await act(async () => {
    release(job);
    await new Promise((resolve) => setTimeout(resolve, 30));
  });

  const copy = texts(renderer.toJSON()).join(" ");
  // The job belongs to the previous interpreter: it must not be shown here,
  // and it must not be cancellable from here either.
  assert.equal(copy.includes(envZh.installing), false);
  assert.equal(copy.includes(envZh.cancelInstall), false);
});

test("a running job is restored for the selected interpreter", async (t) => {
  const job = {
    id: "install-7",
    phase: "running",
    pythonPath: SUGGESTED,
    command: "pip install reme-ai[core]",
    startedAt: new Date().toISOString(),
    logPath: "C:\\Users\\test\\.reme\\install-7.log",
    logTail: "Collecting reme-ai",
  };
  const { renderer, calls, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/installStatus": (input) =>
          input?.jobId || input?.pythonPath ? job : null,
      },
    },
  );
  t.after(restore);

  assert.ok(
    texts(renderer.toJSON()).join(" ").includes(envZh.installing),
    "the running job is shown again",
  );
  // The Host is asked for the job of this interpreter, not for the newest job
  // overall: another interpreter may be installing at the same time.
  const lookups = calls.rpc.filter(
    (call) => call.method === "remeEnv/installStatus",
  );
  assert.ok(lookups.length > 0);
  assert.equal(lookups[0].input.pythonPath, SUGGESTED);
});

test("the LLM test waits for the backend that is saved on disk", async (t) => {
  const { renderer, restore } = await mount();
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  // The saved backend is the one the probe would use.
  assert.equal(
    buttonByText(renderer, envZh.testLlmButton).props.disabled,
    false,
  );

  // Picking another backend in the form does not change the file on disk, so
  // the test would probe the old one.
  const backend = renderer.root
    .findAll((node) => node.type === "select")
    .find((node) => node.props.value === "dashscope");
  assert.ok(backend, "the LLM backend select");
  await act(async () => {
    backend.props.onChange({ target: { value: "anthropic" } });
  });
  assert.ok(
    texts(renderer.toJSON()).join(" ").includes(envZh.testLlmSaveFirst),
    "the button explains that the .env has to be saved first",
  );
  assert.equal(
    buttonByText(renderer, envZh.testLlmButton).props.disabled,
    true,
  );
});

test("the LLM test waits for every unsaved probe setting", async (t) => {
  // The probe reads the whole .env from disk, so a changed model or key is
  // just as untested as a changed backend.
  const { renderer, restore } = await mount();
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  const model = renderer.root.findAll(
    (node) => node.type === "input" && node.props.value === "qwen3.7-plus",
  );
  assert.ok(model.length > 0, "the model field");
  await act(async () => {
    model[0].props.onChange({ target: { value: "another-model" } });
  });
  assert.equal(
    buttonByText(renderer, envZh.testLlmButton).props.disabled,
    true,
    "an unsaved model is not the model the probe would use",
  );

  // A typed key is not on disk either.
  const fresh = await mount();
  try {
    await act(async () => {
      buttonByText(fresh.renderer, envZh.expand).props.onClick();
    });
    const key = fresh.renderer.root.findAll(
      (node) => node.type === "input" && node.props.type === "password",
    )[0];
    await act(async () => {
      key.props.onChange({ target: { value: "sk-typed" } });
    });
    assert.equal(
      buttonByText(fresh.renderer, envZh.testLlmButton).props.disabled,
      true,
      "an unsaved key is not the key the probe would use",
    );
  } finally {
    fresh.restore();
  }
});

test("picking the default backend is not treated as a change", async (t) => {
  // The .env names no backend, so ReMe and the probe both use the default;
  // choosing it in the form changes nothing.
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/readEnv": {
          workspaceDir: "C:\\Users\\test\\.reme",
          exists: true,
          envPath: "C:\\Users\\test\\.reme\\.env",
          entries: [
            {
              key: "LLM_MODEL_NAME",
              value: "qwen3.7-plus",
              masked: "qwen3.7-plus",
              secret: false,
              set: true,
            },
          ],
        },
      },
    },
  );
  t.after(restore);
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  const backend = renderer.root
    .findAll((node) => node.type === "select")
    // The LLM backend is the only select that offers the native providers;
    // the mirror and the embedding backend have their own options.
    .find((node) =>
      node.props.children
        .flat(Infinity)
        .some((option) => option?.props?.value === "anthropic"),
    );
  assert.ok(backend, "the LLM backend select");
  await act(async () => {
    backend.props.onChange({ target: { value: "openai" } });
  });
  assert.equal(
    buttonByText(renderer, envZh.testLlmButton).props.disabled,
    false,
    "the default backend is what the probe already uses",
  );
});

test("a restored job is matched by the interpreter's own path semantics", async (t) => {
  // The Host reports the interpreter with its own spelling; a Windows path
  // folds case and separators, so the job is still the one on screen.
  const job = {
    id: "install-8",
    phase: "running",
    pythonPath: SUGGESTED.replaceAll("\\", "/").toLowerCase(),
    command: "pip install reme-ai[core]",
    startedAt: new Date().toISOString(),
    logPath: "C:\\Users\\test\\.reme\\install-8.log",
    logTail: "",
  };
  const { renderer, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/installStatus": (input) =>
          input?.jobId || input?.pythonPath ? job : null,
      },
    },
  );
  t.after(restore);

  assert.ok(
    texts(renderer.toJSON()).join(" ").includes(envZh.installing),
    "the running job is shown again",
  );
});

test("Forget targets the record the Host named, not the endpoint on screen", async (t) => {
  const CARD = "http://127.0.0.1:2444";
  const RECORD = "http://127.0.0.1:2333";
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: CARD,
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, calls, restore } = await mount(
    { endpoint: CARD },
    {
      responses: {
        "remeEnv/serviceStatus": stopped,
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "The recorded service could not be verified",
          code: "unverified-record",
          pid: 6000,
          record: { endpoint: RECORD, pid: 6000 },
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
        "remeEnv/serviceForget": { forgotten: true, pid: 6000 },
      },
    },
  );
  t.after(restore);

  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  // A status read only covers the endpoint on screen, so it must not drop the
  // record that belongs to the other endpoint. The service card owns the last
  // "Check" button; the dependency card has one with the same label.
  const rechecks = renderer.root.findAll(
    (node) =>
      node.type === "button" && textOf(node.props.children) === envZh.recheck,
  );
  await act(async () => {
    rechecks.at(-1).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  assert.ok(
    buttonByText(renderer, envZh.forgetRecord),
    "the record-only way out survives a status read",
  );
  // The confirmation names the record that blocks the start.
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  const confirming = texts(renderer.toJSON()).join(" ");
  assert.ok(
    confirming.includes(RECORD),
    "the confirmation names the blocked record",
  );
  // It names the process the record describes, so the user sees what they are
  // about to delete.
  assert.ok(confirming.includes("6000"), "the confirmation names the pid");
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });

  const forget = calls.rpc.find(
    (call) => call.method === "remeEnv/serviceForget",
  );
  assert.ok(forget, "forgetting reaches the Host");
  assert.equal(
    forget.input.endpoint,
    RECORD,
    "the record's own endpoint is used, not the endpoint on screen",
  );
  assert.equal(forget.input.expectPid, 6000, "the record is pinned by its pid");
});

test("a record that changed is not forgotten on a second click", async (t) => {
  const stopped = {
    phase: "stopped",
    managed: false,
    endpoint: "http://127.0.0.1:2333",
    logPath: "C:\\Users\\test\\.reme\\service.log",
  };
  const { renderer, calls, restore } = await mount(
    {},
    {
      responses: {
        "remeEnv/serviceStatus": stopped,
        "remeEnv/serviceStart": {
          started: false,
          alreadyRunning: false,
          error: "The recorded service identity could not be verified",
          code: "unverified-record",
          pid: 6000,
          record: { endpoint: "http://127.0.0.1:2333", pid: 6000 },
          command: "",
          logPath: stopped.logPath,
          startedAt: new Date().toISOString(),
        },
        // The record was replaced between the offer and the confirmation.
        "remeEnv/serviceForget": {
          forgotten: false,
          reason: "The recorded service changed since it was inspected",
          code: "record-changed",
          pid: 7000,
          record: { endpoint: "http://127.0.0.1:2333", pid: 7000 },
        },
      },
    },
  );
  t.after(restore);

  await act(async () => {
    buttonByText(renderer, envZh.startService).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  await act(async () => {
    buttonByText(renderer, envZh.forgetRecord).props.onClick();
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 30));
  });

  const copy = texts(renderer.toJSON()).join(" ");
  assert.ok(copy.includes(envZh.stopRecordChanged), "the refusal is explained");
  // The offer is withdrawn: the record the user confirmed is not the one that
  // is there now, so a second click must not be able to delete it.
  assert.equal(
    renderer.root.findAll(
      (node) =>
        node.type === "button" &&
        textOf(node.props.children) === envZh.forgetRecord,
    ).length,
    0,
  );
  assert.equal(
    calls.rpc.filter((call) => call.method === "remeEnv/serviceForget").length,
    1,
    "the withdrawn offer sends no second request",
  );
});
