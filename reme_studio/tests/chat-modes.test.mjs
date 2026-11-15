import assert from "node:assert/strict";
import test from "node:test";
import { isChatAvailable } from "../app/chat-modes.ts";

const job = (backend) => ({ backend, enabled: true, enable_serve: true });
const ordinary = { service: {}, jobs: { chat: job("stream") } };
const proactive = {
  service: {},
  jobs: {
    pa_chat: job("stream"),
    pa_session: job("base"),
    pa_new_session: job("base"),
  },
};

test("chat modes follow exposed jobs, independently of configuration names", () => {
  assert.equal(isChatAvailable(undefined, "chat"), false);
  assert.equal(isChatAvailable(ordinary, "chat"), true);
  assert.equal(isChatAvailable(ordinary, "pa"), false);
  assert.equal(isChatAvailable(proactive, "chat"), false);
  assert.equal(isChatAvailable(proactive, "pa"), true);
  const combined = {
    service: {},
    jobs: { ...ordinary.jobs, ...proactive.jobs },
  };
  assert.equal(isChatAvailable(combined, "chat"), true);
  assert.equal(isChatAvailable(combined, "pa"), true);
});

test("incomplete, disabled, private, or service-excluded PA jobs disable the entry", () => {
  for (const name of Object.keys(proactive.jobs)) {
    for (const change of [
      undefined,
      { ...proactive.jobs[name], enabled: false },
      { ...proactive.jobs[name], enable_serve: false },
      { ...proactive.jobs[name], backend: "background" },
    ]) {
      assert.equal(
        isChatAvailable(
          { ...proactive, jobs: { ...proactive.jobs, [name]: change } },
          "pa",
        ),
        false,
      );
    }
    assert.equal(
      isChatAvailable(
        {
          ...proactive,
          service: {
            jobs: Object.keys(proactive.jobs).filter((key) => key !== name),
          },
        },
        "pa",
      ),
      false,
    );
  }
  assert.equal(
    isChatAvailable({ ...ordinary, service: { jobs: [] } }, "chat"),
    false,
  );
});
