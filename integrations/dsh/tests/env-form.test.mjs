import assert from "node:assert/strict";
import test from "node:test";

import React from "react";
import { act, create } from "react-test-renderer";

import {
  ENV_GROUPS,
  WorkspaceSection,
} from "../dist/environment/client/workspace.js";
import { envZh, fill } from "../dist/environment/client/locales.js";
import { MANAGED_KEYS } from "../dist/environment/env-fields.js";

globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const WORKSPACE = "C:\\Users\\test\\.reme";
const ENV_PATH = `${WORKSPACE}\\.env`;

/** The summary the Host answers a read with: every managed key, in order. */
const SUMMARY = {
  workspaceDir: WORKSPACE,
  exists: true,
  envPath: ENV_PATH,
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
    {
      key: "LLM_API_KEY",
      value: "",
      masked: "••••cret",
      secret: true,
      set: true,
    },
    {
      key: "LLM_BASE_URL",
      value: "https://example.com/v1",
      masked: "https://example.com/v1",
      secret: false,
      set: true,
    },
    {
      key: "EMBEDDING_BACKEND",
      value: "openai",
      masked: "openai",
      secret: false,
      set: true,
    },
    {
      key: "EMBEDDING_MODEL_NAME",
      value: "text-embedding-v4",
      masked: "text-embedding-v4",
      secret: false,
      set: true,
    },
    {
      key: "EMBEDDING_API_KEY",
      value: "",
      masked: "••••MPTY",
      secret: true,
      set: true,
    },
    {
      key: "EMBEDDING_BASE_URL",
      value: "",
      masked: "",
      secret: false,
      set: false,
    },
  ],
};

const t = (key) => envZh[key];

function textOf(value) {
  if (Array.isArray(value)) return value.map(textOf).join("");
  return typeof value === "string" ? value : "";
}

/** The `.env` form, driven by the real section and its own workspace hook. */
function Harness({ rpc, workspaceDir = WORKSPACE }) {
  return React.createElement(WorkspaceSection, {
    // The tab mounts the section with the workspace as its key, so a switch
    // disposes the previous workspace's state.
    key: workspaceDir,
    t,
    rpc,
    workspaceDir,
    pythonPath: "C:\\env\\python.exe",
    writable: true,
    onWorkspaceDir: async () => true,
  });
}

async function mount(options = {}) {
  const calls = { rpc: [] };
  let resolveWrite;
  const pending = options.pendingWrite
    ? new Promise((resolve) => {
        resolveWrite = resolve;
      })
    : undefined;
  const rpc = {
    call: async (_path, method, payload) => {
      calls.rpc.push({ method, input: payload?.args?.input });
      if (method === "remeEnv/readEnv") return { ok: true, value: SUMMARY };
      if (method === "remeEnv/writeEnv") {
        if (pending) return { ok: true, value: await pending };
        return { ok: true, value: SUMMARY };
      }
      return { ok: true, value: null };
    },
  };

  let renderer;
  await act(async () => {
    renderer = create(React.createElement(Harness, { rpc }));
  });
  // The card starts collapsed; the form is behind the expand button.
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  // Wait for the summary to arrive instead of guessing how long it takes.
  await waitFor(() => fieldValue(renderer, "model") === "qwen3.7-plus");
  return {
    renderer,
    calls,
    resolveWrite(value = SUMMARY) {
      resolveWrite(value);
    },
    restore: () => {
      act(() => {
        renderer.unmount();
      });
    },
  };
}

/** Let the effects settle until a condition holds, or the wait runs out. */
async function waitFor(check, tries = 100) {
  for (let attempt = 0; attempt < tries; attempt += 1) {
    if (check()) return;
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 10));
    });
  }
  assert.ok(check(), "the form did not reach the expected state");
}

function buttonByText(renderer, label) {
  const found = renderer.root.findAll(
    (node) => node.type === "button" && textOf(node.props.children) === label,
  );
  assert.ok(found.length > 0, `button not found: ${label}`);
  return found[0];
}

/**
 * The `.env` controls, in the order the form renders them: the two backend
 * selects are not inputs, so the six inputs are the two groups' text and
 * secret fields.
 */
const FIELD = {
  model: 0,
  apiKey: 1,
  baseUrl: 2,
  embeddingModel: 3,
  embeddingApiKey: 4,
  embeddingBaseUrl: 5,
};

/** The inputs of the `.env` form, without the workspace directory field. */
function envInputs(renderer) {
  const container = renderer.root.findAll((node) =>
    String(node.props.className ?? "").includes("reme-env-config"),
  );
  assert.equal(container.length, 1, "the .env form is rendered once");
  return container[0].findAll((node) => node.type === "input");
}

function fieldInput(renderer, name) {
  return envInputs(renderer)[FIELD[name]];
}

/** The value a `.env` field shows right now. */
function fieldValue(renderer, name) {
  return fieldInput(renderer, name)?.props.value ?? "";
}

/** The save's own report, or an empty string while there is none. */
function statusText(renderer) {
  const found = renderer.root.findAll((node) =>
    String(node.props.className ?? "").includes("reme-settings-success"),
  );
  if (!found.length) return "";
  return textOf(found[0].props.children);
}

/** Every success message on screen, in render order. */
function successTexts(renderer) {
  return renderer.root
    .findAll((node) =>
      String(node.props.className ?? "").includes("reme-settings-success"),
    )
    .map((node) => textOf(node.props.children));
}

function writeCall(calls) {
  const write = calls.rpc
    .filter((call) => call.method === "remeEnv/writeEnv")
    .at(-1);
  assert.ok(write, "the save reached the Host");
  return write.input;
}

test("the form shows every managed key, in the shared order", async (t) => {
  const groups = ENV_GROUPS;
  assert.deepEqual(
    groups.flatMap((group) => group.fields.map((field) => field.key)),
    [...MANAGED_KEYS],
  );
  // The secret flag is the shared one, not a second copy of it.
  assert.deepEqual(
    groups
      .flatMap((group) => group.fields)
      .filter((field) => field.kind === "secret")
      .map((field) => field.key),
    ["LLM_API_KEY", "EMBEDDING_API_KEY"],
  );

  const { renderer, restore } = await mount();
  t.after(restore);
  const inputs = envInputs(renderer);
  assert.equal(inputs.length, 6);
  assert.equal(inputs[FIELD.apiKey].props.type, "password");
  assert.equal(inputs[FIELD.embeddingApiKey].props.type, "password");
  assert.equal(fieldInput(renderer, "model").props.value, "qwen3.7-plus");
});

test("an edit typed while the save is in flight survives the save", async (t) => {
  const { renderer, calls, resolveWrite, restore } = await mount({
    pendingWrite: true,
  });
  t.after(restore);

  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-b" },
    });
  });
  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  // The save is in flight and the field stays editable.
  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-c" },
    });
  });
  await act(async () => {
    resolveWrite();
  });
  await waitFor(() => statusText(renderer) !== "");

  // The save carried the value that was submitted...
  assert.equal(writeCall(calls).values.LLM_MODEL_NAME, "model-b");
  // ...and the newer edit is still on screen instead of being reverted.
  assert.equal(fieldValue(renderer, "model"), "model-c");
});

test("a blank secret keeps the stored credential, a blank field clears it", async (t) => {
  const { renderer, calls, restore } = await mount();
  t.after(restore);

  // A replacement key is typed and cleared again: "leave blank to keep".
  await act(async () => {
    fieldInput(renderer, "apiKey").props.onChange({
      target: { value: "sk-new" },
    });
  });
  await act(async () => {
    fieldInput(renderer, "apiKey").props.onChange({ target: { value: "" } });
  });
  // A non-secret field is cleared explicitly.
  await act(async () => {
    fieldInput(renderer, "baseUrl").props.onChange({ target: { value: "" } });
  });
  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  await waitFor(() => statusText(renderer) !== "");

  const values = writeCall(calls).values;
  // The emptied secret does not travel: the stored credential survives.
  assert.equal("LLM_API_KEY" in values, false);
  // The emptied field is an explicit clear.
  assert.equal(values.LLM_BASE_URL, "");
  // Nothing else travelled either.
  assert.equal("LLM_MODEL_NAME" in values, false);
  assert.equal("EMBEDDING_API_KEY" in values, false);
});

test("the saved message admits that newer edits are still pending", async (t) => {
  const { renderer, resolveWrite, restore } = await mount({
    pendingWrite: true,
  });
  t.after(restore);

  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-b" },
    });
  });
  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-c" },
    });
  });
  await act(async () => {
    resolveWrite();
  });
  await waitFor(() => statusText(renderer) !== "");

  // The form is not clean, so the message must not read like a clean save.
  assert.equal(
    statusText(renderer),
    fill(envZh.savedEnvPending, { path: ENV_PATH }),
  );
  assert.notEqual(
    statusText(renderer),
    fill(envZh.savedEnv, { path: ENV_PATH }),
  );
});

test("a save without newer edits reports a clean form", async (t) => {
  const { renderer, restore } = await mount();
  t.after(restore);

  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-b" },
    });
  });
  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  await waitFor(() => statusText(renderer) !== "");

  assert.equal(statusText(renderer), fill(envZh.savedEnv, { path: ENV_PATH }));
});

const OTHER_WORKSPACE = "D:\\Other\\.reme";
const OTHER_ENV_PATH = `${OTHER_WORKSPACE}\\.env`;

/** The summary of another workspace, so a switch is visible in the form. */
const OTHER_SUMMARY = {
  ...SUMMARY,
  workspaceDir: OTHER_WORKSPACE,
  envPath: OTHER_ENV_PATH,
  entries: SUMMARY.entries.map((entry) =>
    entry.key === "LLM_MODEL_NAME"
      ? { ...entry, value: "other-model", masked: "other-model" }
      : entry,
  ),
};

/** A Host whose `.env` reads answer for the workspace they were asked about. */
function switchingRpc() {
  return {
    call: async (_path, method, payload) => {
      const input = payload?.args?.input ?? {};
      if (method === "remeEnv/readEnv")
        return {
          ok: true,
          value:
            input.workspaceDir === OTHER_WORKSPACE ? OTHER_SUMMARY : SUMMARY,
        };
      if (method === "remeEnv/writeEnv") return { ok: true, value: SUMMARY };
      if (method === "remeEnv/testLlm")
        return {
          ok: true,
          value: { ok: true, status: 200, latencyMs: 12, error: "" },
        };
      return { ok: true, value: null };
    },
  };
}

test("a workspace switch drops the completed save and LLM-test results", async (t) => {
  const rpc = switchingRpc();
  let renderer;
  await act(async () => {
    renderer = create(React.createElement(Harness, { rpc }));
  });
  t.after(() => act(() => renderer.unmount()));
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  await waitFor(() => fieldValue(renderer, "model") === "qwen3.7-plus");

  // A completed save and a successful probe, both for the first workspace.
  await act(async () => {
    fieldInput(renderer, "model").props.onChange({
      target: { value: "model-b" },
    });
  });
  await act(async () => {
    buttonByText(renderer, envZh.saveEnv).props.onClick();
  });
  await waitFor(() =>
    successTexts(renderer).includes(fill(envZh.savedEnv, { path: ENV_PATH })),
  );
  await act(async () => {
    buttonByText(renderer, envZh.testLlmButton).props.onClick();
  });
  await waitFor(() =>
    successTexts(renderer).includes(fill(envZh.llmOk, { ms: 12 })),
  );

  // The tab remounts the section with the new workspace as its key.
  await act(async () => {
    renderer.update(
      React.createElement(Harness, { rpc, workspaceDir: OTHER_WORKSPACE }),
    );
  });
  await act(async () => {
    buttonByText(renderer, envZh.expand).props.onClick();
  });
  await waitFor(() => fieldValue(renderer, "model") === "other-model");

  // Neither result belonged to this workspace, so neither is shown.
  assert.deepEqual(successTexts(renderer), []);
  // The form shows this workspace's file, not the previous workspace's.
  const shown = (text) =>
    renderer.root.findAll((node) => textOf(node.props.children).includes(text))
      .length;
  assert.ok(shown(OTHER_ENV_PATH) > 0, "the new .env path is on screen");
  assert.equal(shown(ENV_PATH), 0, "the previous .env path is gone");
});
