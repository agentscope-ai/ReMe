import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import test from "node:test";
import vm from "node:vm";
import React from "react";
import { act, create } from "react-test-renderer";
import { resolveConfig } from "../dist/config.js";

/**
 * The snapshot store the form model binds to.
 *
 * `@deepseek-ai/dsh-client-store` is not installed in this package, so the
 * harness supplies the contract the real package offers: the store keeps the
 * value it was given and hands that very reference back from `getSnapshot()`,
 * and it notifies its listeners only when the value actually changed. The
 * browser loads the real package; this stand-in only has to behave like it for
 * the card's reads.
 */
function createSnapshotStore(init) {
  let value = init;
  const listeners = new Set();
  return {
    getSnapshot: () => value,
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    set(next) {
      if (Object.is(next, value)) return;
      value = next;
      for (const listener of [...listeners]) listener();
    },
  };
}

/**
 * Evaluate the SettingsFormModel this plugin links against, straight out of the
 * shipped primitives bundle.
 *
 * The package cannot be imported in Node (it loads CSS modules, icons and the
 * syntax highlighter), but the form model region is plain JavaScript. Running
 * that region means the card is exercised against DSH's real model instead of a
 * re-implementation of it; a changed bundle layout fails here loudly.
 */
function loadSettingsFormModel() {
  // Resolve the installed package rather than a fixed path: the packaged-plugin
  // check installs it hoisted above this package, while the repository keeps it
  // inside the package's own node_modules.
  const manifest = createRequire(import.meta.url).resolve(
    "@deepseek-ai/dsh-client-ui-primitives/package.json",
  );
  const source = readFileSync(
    join(dirname(manifest), "lib", "index.js"),
    "utf8",
  );
  const start = source.indexOf(
    "//#region lib/types/settings-form/form-model.js",
  );
  const end = source.indexOf("//#endregion", start + 1);
  assert.ok(
    start >= 0 && end > start,
    "unexpected @deepseek-ai/dsh-client-ui-primitives bundle layout",
  );
  // The model binds its projections through `createSnapshotStore`, which the
  // browser gets from `@deepseek-ai/dsh-client-store`; the harness passes the
  // stand-in above instead.
  return new Function(
    "createSnapshotStore",
    `${source.slice(start, end)}\nreturn SettingsFormModel;`,
  )(createSnapshotStore);
}

async function editor(mutate) {
  let module;
  const bundle = await readFile(
    new URL("../dist/client.js", import.meta.url),
    "utf8",
  );
  vm.runInNewContext(bundle, {
    URL,
    // The card disposes its form model on a deferred timer, so the new
    // context needs the timer globals: without them the cleanup throws
    // `ReferenceError: setTimeout is not defined` and React reports it.
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
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
    name === "@deepseek-ai/dsh-client-ui-primitives"
      ? { SettingsFormModel: loadSettingsFormModel() }
      : nodeRequire(name),
  );
  let Component;
  let snapshot = {
    status: "ready",
    value: resolveConfig({}, {}),
    base: resolveConfig({}, {}),
    // The Host reports the user layer it stored; every field here is an
    // override, which is what "reset to defaults" clears.
    user: resolveConfig({}, {}),
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
    input(index) {
      return renderer.root.findAllByType("input")[index].props;
    },
    button(label) {
      return renderer.root
        .findAllByType("button")
        .find((button) => button.children.includes(label));
    },
    text: () => JSON.stringify(renderer.toJSON()),
    async close() {
      act(() => renderer.unmount());
      // The card defers its disposal to the next tick, so the cleanup runs
      // inside the test instead of after it.
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
      });
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
    await ui.close();
  }
});

test("repaints the controls and the unsaved marker from the model's publishes", async () => {
  const ui = await editor(async () => true);
  try {
    assert.doesNotMatch(ui.text(), /unsaved/);
    // A staged edit publishes, and the card renders what the model holds: the
    // control shows the draft and the card says the form is unsaved.
    ui.change(0, "http://memory.test");
    assert.equal(ui.input(0).value, "http://memory.test");
    assert.match(ui.text(), /unsaved/);
    // Dropping the drafts publishes as well, and the marker goes with them.
    act(() => ui.button("discard").props.onClick());
    assert.doesNotMatch(ui.text(), /unsaved/);
    assert.notEqual(ui.input(0).value, "http://memory.test");
  } finally {
    await ui.close();
  }
});

test("follows the namespace status through the model's publishes", async () => {
  const ui = await editor(async () => true);
  try {
    assert.notEqual(ui.text(), "null");
    // The card reads the scope through the model, so a status change reaches
    // it as a publish rather than as a subscription of its own.
    act(() => ui.update({ status: "loading" }));
    assert.match(ui.text(), /loading/);
    act(() => ui.update({ status: "unavailable" }));
    assert.equal(ui.text(), "null");
  } finally {
    await ui.close();
  }
});

test("the save button follows the model's own saving state", async () => {
  let release = () => {};
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const ui = await editor(async () => {
    await gate;
    return true;
  });
  try {
    ui.change(0, "http://memory.test");
    await act(async () => {
      ui.button("save").props.onClick();
    });
    // The write is in flight: the model's own state says so, and no second
    // flag in the card could disagree with it.
    const inFlight = ui.button("saving");
    assert.ok(inFlight);
    assert.equal(inFlight.props.disabled, true);
    assert.equal(ui.button("save"), undefined);
    assert.equal(ui.calls.length, 1);

    await act(async () => {
      release();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    // The save landed: the button is idle again, disabled because the form
    // holds no drafts, and the result is reported.
    assert.equal(ui.button("saving"), undefined);
    assert.match(ui.text(), /"saved"/);
    assert.equal(ui.button("save").props.disabled, true);
  } finally {
    await ui.close();
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
      await ui.close();
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
    await ui.close();
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
    await ui.close();
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
        await ui.close();
      }
    },
  );
}
