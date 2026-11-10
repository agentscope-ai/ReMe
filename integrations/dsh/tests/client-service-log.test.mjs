import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import { act, create } from "react-test-renderer";
import { useService } from "../dist/environment/client/service.js";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;
function deferred() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("dismissed or old-workspace log requests cannot reopen the current log panel", async (t) => {
  let pending = deferred();
  let current;
  let signal;
  const rpc = {
    async call(_path, method, _args, active) {
      assert.equal(method, "remeEnv/serviceLog");
      signal = active;
      return { ok: true, value: await pending.promise };
    },
  };
  function Harness({ directory }) {
    current = useService(rpc, {
      t: (key) => key,
      endpoint: "http://localhost:2333",
      local: false,
      pythonPath: "/python",
      workspaceDir: directory,
      authToken: "",
    });
    return null;
  }
  let renderer;
  await act(async () => {
    renderer = create(React.createElement(Harness, { directory: "/old" }));
  });
  t.after(async () => {
    pending.resolve({ lines: "", logPath: "" });
    await act(async () => renderer.unmount());
  });
  act(() => {
    void current.viewLog();
  });
  act(() => current.dismissLog());
  assert.equal(signal.aborted, true);
  await act(async () =>
    pending.resolve({ lines: "dismissed response", logPath: "/old/log" }),
  );
  assert.equal(current.log.open, false);

  pending = deferred();
  act(() => {
    void current.viewLog();
  });
  await act(async () =>
    renderer.update(React.createElement(Harness, { directory: "/new" })),
  );
  assert.equal(signal.aborted, true);
  await act(async () =>
    pending.resolve({ lines: "old workspace response", logPath: "/old/log" }),
  );
  assert.equal(current.log.open, false);
  assert.equal(current.log.lines, "");
});
