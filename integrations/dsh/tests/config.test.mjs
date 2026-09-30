import assert from "node:assert/strict";
import test from "node:test";
import { Context } from "@deepseek-ai/cordis";
import Loader from "@deepseek-ai/cordis-plugin-loader";
import z from "@deepseek-ai/schemastery";
import { Config, resolveConfig } from "../dist/config.js";

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
  assert.equal(valid.value.autoMemoryInterval.get(), 5);
  assert.equal(valid.value.shutdownTimeoutMs.get(), 5000);
  assert.equal(resolveConfig(valid.value).language, "zh");

  for (const input of [
    { endpoint: "not-a-url" },
    { endpoint: "http://localhost:bad-port" },
    { dreamCron: "every night" },
    { timezone: "Mars/Olympus" },
  ]) {
    assert.ok(Config["~standard"].validate(input).issues?.length);
  }
});

test("volatile settings survive Host projection and browser schema rehydration", () => {
  const fields = Object.entries(Config.dict).flatMap(([name, schema]) => {
    if (!schema.meta.volatile) return [];
    const plain = new z(schema.toJSON());
    delete plain.meta.volatile;
    return [[name, plain]];
  });
  const hostForm = z.object(Object.fromEntries(fields));
  const browserForm = new z(JSON.parse(JSON.stringify(hostForm.toJSON())));
  const defaults = Config["~standard"].validate({}).value;
  const values = Object.fromEntries(
    fields.map(([name]) => [name, defaults[name].get()]),
  );
  assert.equal(browserForm["~standard"].validate(values).issues, undefined);
  assert.ok(
    browserForm["~standard"].validate({ ...values, dreamCron: "bad" }).issues
      ?.length,
  );
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
  assert.throws(
    () => resolveConfig({ apiKey: "unsupported" }, {}),
    /Unknown ReMe config option/,
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

test("rejects settings that cannot be scheduled or reached", () => {
  assert.throws(
    () => resolveConfig({ endpoint: "file:///tmp/reme" }),
    /absolute http/,
  );
  assert.throws(
    () => resolveConfig({ dreamCron: "every night" }),
    /daily form/,
  );
});

test("Loader keeps the previous live values when an update fails schema validation", async () => {
  const ctx = new Context();
  let live;
  try {
    await ctx.plugin(Loader);
    ctx.loader.builtins.remeTest = {
      name: "reme-test",
      Config,
      apply(_owner, input) {
        live = input;
      },
    };
    const id = await ctx.loader.create({
      id: "reme-memory",
      name: "cordis:remeTest",
      config: { endpoint: "http://valid.test", autoDreamEnabled: false },
    });
    const entry = ctx.loader.resolve(id);
    await entry.fiber.await();
    for (const patch of [
      { endpoint: "not-a-url" },
      { dreamCron: "every night" },
      { timezone: "Mars/Olympus" },
    ]) {
      await entry.update({
        config: {
          endpoint: "http://valid.test",
          autoDreamEnabled: false,
          ...patch,
        },
      });
      assert.equal(live.endpoint.get(), "http://valid.test");
      assert.equal(live.dreamCron.get(), "0 23 * * *");
      assert.equal(live.timezone.get(), "Asia/Shanghai");
    }
    await entry.update({
      config: { endpoint: "http://updated.test", autoDreamEnabled: false },
    });
    assert.equal(live.endpoint.get(), "http://updated.test");
  } finally {
    await ctx.fiber.dispose();
  }
});
