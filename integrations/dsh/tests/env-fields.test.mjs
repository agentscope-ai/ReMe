import assert from "node:assert/strict";
import test from "node:test";

import {
  ENV_FIELDS,
  envFieldMeta,
  envUpdates,
  MANAGED_KEYS,
  SECRET_KEYS,
  storedState,
} from "../dist/environment/env-fields.js";

/** A file that already holds a value for every key that needs one. */
const STORED = { EMBEDDING_API_KEY: true };

test("the managed keys are the shared list, in one order", () => {
  assert.deepEqual(MANAGED_KEYS, [
    "LLM_BACKEND",
    "LLM_MODEL_NAME",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
  ]);
  assert.deepEqual(
    [...MANAGED_KEYS],
    ENV_FIELDS.map((field) => field.key),
  );
});

test("only the two credentials are secret", () => {
  assert.deepEqual([...SECRET_KEYS].sort(), [
    "EMBEDDING_API_KEY",
    "LLM_API_KEY",
  ]);
  for (const key of MANAGED_KEYS)
    assert.equal(SECRET_KEYS.has(key), key.endsWith("API_KEY"), key);
});

test("the metadata of a key carries its flag and its empty rule", () => {
  assert.equal(envFieldMeta("LLM_API_KEY")?.secret, true);
  assert.equal(envFieldMeta("LLM_API_KEY")?.empty, "keep");
  assert.equal(envFieldMeta("LLM_BASE_URL")?.secret, false);
  assert.equal(envFieldMeta("LLM_BASE_URL")?.empty, "clear");
  assert.equal(envFieldMeta("EMBEDDING_API_KEY")?.disabledValue, "EMPTY");
  assert.equal(envFieldMeta("NOT_MANAGED"), undefined);
});

test("an untouched key does not travel", () => {
  const updates = envUpdates({ LLM_MODEL_NAME: "qwen3.7-plus" }, STORED);
  assert.deepEqual(updates, { LLM_MODEL_NAME: "qwen3.7-plus" });
  // The stored value of every other key survives the save.
  assert.equal("LLM_API_KEY" in updates, false);
  assert.equal("LLM_BASE_URL" in updates, false);
  assert.equal("EMBEDDING_API_KEY" in updates, false);
});

test("a blank secret is dropped, so the stored credential survives", () => {
  // The key was typed and cleared again: the draft holds an empty value, but
  // an empty secret means "keep what is stored".
  const updates = envUpdates(
    { LLM_API_KEY: "", EMBEDDING_API_KEY: "" },
    { LLM_API_KEY: true, EMBEDDING_API_KEY: true },
  );
  assert.equal("LLM_API_KEY" in updates, false);
  assert.equal("EMBEDDING_API_KEY" in updates, false);
  assert.deepEqual(updates, {});
});

test("a blank non-secret is an explicit clear", () => {
  const updates = envUpdates({ LLM_BASE_URL: "" }, STORED);
  assert.deepEqual(updates, { LLM_BASE_URL: "" });
});

test("a typed secret travels", () => {
  const updates = envUpdates({ LLM_API_KEY: "sk-new" }, STORED);
  assert.deepEqual(updates, { LLM_API_KEY: "sk-new" });
});

test("the embedding key is written as EMPTY while it is unset", () => {
  // ReMe 0.4.0.x instantiates the embedding component unconditionally, so an
  // empty key there aborts the service start.
  const untouched = envUpdates({ LLM_MODEL_NAME: "qwen3.7-plus" }, {});
  assert.equal(untouched.EMBEDDING_API_KEY, "EMPTY");
  // A cleared embedding key is the same case: it must not stay empty.
  const cleared = envUpdates(
    { EMBEDDING_API_KEY: "" },
    { EMBEDDING_API_KEY: false },
  );
  assert.equal(cleared.EMBEDDING_API_KEY, "EMPTY");
  // A stored key is never replaced by the placeholder.
  const stored = envUpdates({}, { EMBEDDING_API_KEY: true });
  assert.equal("EMBEDDING_API_KEY" in stored, false);
});

test("the stored state reports which keys hold a value", () => {
  // The tab reads this from the masked projection, so a set secret counts as
  // present through its non-empty mask.
  const state = storedState({
    LLM_MODEL_NAME: "qwen3.7-plus",
    LLM_API_KEY: "",
  });
  assert.equal(state.LLM_MODEL_NAME, true);
  assert.equal(state.LLM_API_KEY, false);
  assert.equal(state.LLM_BASE_URL, false);
  assert.deepEqual(
    Object.keys(state).sort(),
    [...MANAGED_KEYS].sort(),
    "every managed key is reported",
  );
});

test("a key the tab does not manage travels unchanged", () => {
  // The Host must not silently drop a value it was asked to write.
  const updates = envUpdates(
    { LLM_MODEL_NAME: "qwen3.7-plus", SECRET_OTHER: "value" },
    STORED,
  );
  assert.deepEqual(updates, {
    LLM_MODEL_NAME: "qwen3.7-plus",
    SECRET_OTHER: "value",
  });
});
