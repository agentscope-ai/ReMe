import assert from "node:assert/strict";
import test from "node:test";
import { isTrustedSender } from "../src/main/ipc.ts";

test("only a live main frame in an owned window may send IPC", () => {
  const frame = { detached: false };
  const sender = { id: 1, isDestroyed: () => false, mainFrame: frame };
  const allowed = (id: number) => id === 1;
  assert.equal(isTrustedSender({ sender, senderFrame: frame }, allowed), true);
  assert.equal(isTrustedSender({ sender, senderFrame: null }, allowed), false);
  assert.equal(
    isTrustedSender({ sender, senderFrame: { detached: false } }, allowed),
    false,
  );
  assert.equal(
    isTrustedSender({ sender, senderFrame: frame }, () => false),
    false,
  );
  frame.detached = true;
  assert.equal(isTrustedSender({ sender, senderFrame: frame }, allowed), false);
});

test("a destroyed or unknown sender is rejected before touching its disposed main frame", () => {
  const sender = {
    id: 1,
    isDestroyed: () => true,
    get mainFrame(): object {
      throw new Error("disposed frame");
    },
  };
  assert.equal(
    isTrustedSender({ sender, senderFrame: null }, () => true),
    false,
  );
  sender.isDestroyed = () => false;
  assert.equal(
    isTrustedSender({ sender, senderFrame: null }, () => false),
    false,
  );
});
