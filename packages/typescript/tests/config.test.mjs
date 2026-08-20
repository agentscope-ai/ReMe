import assert from "node:assert/strict";
import test from "node:test";
import {
  Config,
  mergeSettings,
  resolveConfig,
  SettingsConfig,
  settingsFrom,
  validateSettings,
} from "../dist/dsh/config.js";

test("resolves the established ReMe host and port environment", () => {
  const config = resolveConfig(
    {},
    { REME_HOST: "memory.local", REME_PORT: "2444" },
  );
  assert.equal(config.endpoint, "http://memory.local:2444");
  assert.equal(config.autoMemoryInterval, 5);
  assert.equal(config.dreamCron, "0 23 * * *");
});

test("exports a Cordis schema that rejects invalid configuration", async () => {
  const result = await Config["~standard"].validate({
    autoMemoryInterval: "five",
  });
  assert.ok(result.issues?.length);

  const valid = await Config["~standard"].validate({ language: "zh" });
  assert.equal(valid.issues, undefined);
  assert.equal(valid.value.autoMemoryInterval, 5);
  assert.equal(valid.value.shutdownTimeoutMs, 5000);
});

test("rejects unknown options and invalid IANA timezones", () => {
  assert.throws(
    () => resolveConfig({ autoMemoryIntervl: 3 }, {}),
    /Unknown ReMe config option/,
  );
  assert.throws(
    () => resolveConfig({ timezone: "Mars/Olympus" }, {}),
    /Invalid ReMe timezone/,
  );
});

test("normalizes bounded plugin configuration", () => {
  const config = resolveConfig(
    {
      endpoint: "http://localhost:2333///",
      language: "zh",
      autoMemoryInterval: 0,
      searchLimit: 100,
      rootAgentsOnly: false,
    },
    {},
  );
  assert.equal(config.endpoint, "http://localhost:2333");
  assert.equal(config.language, "zh");
  assert.equal(config.autoMemoryInterval, 1);
  assert.equal(config.searchLimit, 50);
  assert.equal(config.rootAgentsOnly, false);
});

test("projects the editable DSH settings without deployment-only secrets", async () => {
  const base = resolveConfig({ apiKey: "secret", dreamIntervalMs: 5000 }, {});
  const settings = settingsFrom(base);
  assert.equal("apiKey" in settings, false);
  assert.equal("dreamIntervalMs" in settings, false);
  const validated = await SettingsConfig["~standard"].validate({
    ...settings,
    searchLimit: 8,
  });
  assert.equal(validated.issues, undefined);
  const merged = mergeSettings(base, validated.value);
  assert.equal(merged.apiKey, "secret");
  assert.equal(merged.searchLimit, 8);
});

test("rejects settings that cannot be scheduled or reached", () => {
  const settings = settingsFrom(resolveConfig({}, {}));
  assert.throws(
    () => validateSettings({ ...settings, endpoint: "file:///tmp/reme" }),
    /absolute http/,
  );
  assert.throws(
    () => validateSettings({ ...settings, dreamCron: "every night" }),
    /daily form/,
  );
});
