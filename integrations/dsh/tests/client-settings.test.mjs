import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import vm from "node:vm";
import React from "react";
import { act, create } from "react-test-renderer";
import { resolveConfig } from "../dist/config.js";

async function editor(mutate) {
  let module;
  const bundle = await readFile(
    new URL("../dist/client.js", import.meta.url),
    "utf8",
  );
  vm.runInNewContext(bundle, {
    URL,
    document: {
      head: { appendChild() {} },
      createElement: () => ({ dataset: {}, remove() {} }),
    },
    window: {
      __ModuleLoader__: {
        load(value) {
          module = value;
        },
      },
    },
  });
  const nodeRequire = createRequire(import.meta.url);
  const client = module.factory((name) =>
    name === "@deepseek-ai/dsh-client-ui-primitives" ? {} : nodeRequire(name),
  );
  let Component;
  let snapshot = {
    status: "ready",
    value: resolveConfig({}, {}),
    base: resolveConfig({}, {}),
    revision: 7,
    writable: true,
  };
  const listeners = new Set();
  const calls = [];
  const scope = {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    async mutate(ops, revision) {
      calls.push({ ops: JSON.parse(JSON.stringify(ops)), revision });
      const accepted = await mutate(ops, revision);
      if (accepted) {
        const value = { ...snapshot.value };
        for (const op of ops)
          value[op.path[0]] =
            op.op === "unset" ? snapshot.base[op.path[0]] : op.value;
        update({ value, revision: snapshot.revision + 1 });
      }
      return accepted;
    },
  };
  function update(next) {
    snapshot = { ...snapshot, ...next };
    for (const listener of listeners) listener();
  }
  const disposers = [];
  client.apply({
    locale: { bind: () => (key) => key, register: () => () => {} },
    configForms: {
      get: () => scope,
      whileServed: (_names, register) => register(),
    },
    slots: {
      inject: (_name, register) => register(),
      register(options, component) {
        if (options.name === "plugins.row.config") Component = component;
        return () => {};
      },
    },
    get: () => ({ rpc: {} }),
    effect(factory) {
      disposers.push(factory());
    },
  });
  let renderer;
  act(() => {
    renderer = create(
      React.createElement(Component, { scope, t: (key) => key, view: "page" }),
    );
  });
  return {
    calls,
    update,
    change(index, value) {
      act(() => {
        const input = renderer.root.findAllByType("input")[index];
        input.props.onChange({ target: { value } });
      });
    },
    button(label) {
      return renderer.root
        .findAllByType("button")
        .find((button) => button.children.includes(label));
    },
    text: () => JSON.stringify(renderer.toJSON()),
    close() {
      act(() => renderer.unmount());
      for (const dispose of disposers) dispose?.();
    },
  };
}

test("saves all edited fields in one revision-fenced mutation", async () => {
  const ui = await editor(async () => true);
  try {
    ui.change(0, "http://memory.test");
    ui.change(1, "8");
    await act(async () => {
      await ui.button("save").props.onClick();
    });
    assert.deepEqual(ui.calls, [
      {
        ops: [
          { op: "set", path: ["endpoint"], value: "http://memory.test" },
          { op: "set", path: ["searchLimit"], value: 8 },
        ],
        revision: 7,
      },
    ]);
    assert.match(ui.text(), /saved/);
    assert.equal(ui.button("save").props.disabled, true);
  } finally {
    ui.close();
  }
});

for (const outcome of ["refused", "transport failure"]) {
  test(`preserves edits after ${outcome} and keeps their original revision`, async () => {
    const ui = await editor(async () => {
      if (outcome === "transport failure") throw new Error("offline");
      return false;
    });
    try {
      ui.change(0, "http://memory.test");
      act(() => ui.update({ revision: 8 }));
      await act(async () => {
        await ui.button("save").props.onClick();
      });
      assert.equal(ui.calls[0].revision, 7);
      assert.match(ui.text(), /saveFailed/);
      assert.match(ui.text(), /http:\/\/memory.test/);
      assert.equal(ui.button("save").props.disabled, false);
      act(() => ui.button("discard").props.onClick());
      assert.doesNotMatch(ui.text(), /http:\/\/memory.test/);
    } finally {
      ui.close();
    }
  });
}

test("resets overrides in one mutation only after Save", async () => {
  const ui = await editor(async () => true);
  try {
    act(() => ui.button("reset").props.onClick());
    assert.equal(ui.calls.length, 0);
    await act(async () => {
      await ui.button("save").props.onClick();
    });
    assert.equal(ui.calls.length, 1);
    assert.equal(ui.calls[0].ops.length, 15);
    assert.ok(ui.calls[0].ops.every((op) => op.op === "unset"));
    assert.equal(ui.calls[0].revision, 7);
  } finally {
    ui.close();
  }
});

test("read-only configuration cannot be saved", async () => {
  const ui = await editor(async () => true);
  try {
    ui.change(0, "http://memory.test");
    act(() => ui.update({ writable: false }));
    await act(async () => {
      await ui.button("save").props.onClick();
    });
    assert.equal(ui.calls.length, 0);
    assert.equal(ui.button("save").props.disabled, true);
    assert.match(ui.text(), /readOnly/);
  } finally {
    ui.close();
  }
});

for (const endpoint of [
  "http://alice:SECRET@localhost:2333",
  "http://localhost:2333?token=SECRET",
  "http://localhost:2333#search",
  "http://localhost:2333?",
  "http://localhost:2333#",
]) {
  test(
    "rejects unsupported endpoint components before saving: " +
      new URL(endpoint).hash +
      new URL(endpoint).search,
    async () => {
      const ui = await editor(async () => true);
      try {
        ui.change(0, endpoint);
        assert.equal(ui.button("save").props.disabled, true);
        await act(async () => {
          await ui.button("save").props.onClick();
        });
        assert.equal(ui.calls.length, 0);
      } finally {
        ui.close();
      }
    },
  );
}
