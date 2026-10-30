import assert from "node:assert/strict";
import test from "node:test";

import {
  SETTINGS_SPECS,
  SETTING_FIELDS,
  timeFromCron,
  validTimezone,
} from "../dist/client/settings-specs.js";

const spec = (field) => {
  const found = SETTINGS_SPECS.find((item) => item.field === field);
  assert.ok(found, `missing spec for ${field}`);
  return found;
};

/** The write a draft stages, or the string "invalid" when it is refused. */
const parse = (field, text) => {
  const write = spec(field).parse(text);
  return write === undefined ? "invalid" : write;
};

test("one spec per setting, in display order", () => {
  assert.deepEqual(
    SETTINGS_SPECS.map((item) => item.field),
    [...SETTING_FIELDS],
  );
  assert.equal(SETTINGS_SPECS.length, 15);
});

test("numbers accept a bounded whole number and clear on empty", () => {
  assert.deepEqual(parse("searchLimit", "8"), { kind: "set", value: 8 });
  assert.deepEqual(parse("searchLimit", " 12 "), { kind: "set", value: 12 });
  assert.deepEqual(parse("searchLimit", ""), { kind: "clear" });
  // Out of range, not whole, or not a number blocks the save.
  assert.equal(parse("searchLimit", "0"), "invalid");
  assert.equal(parse("searchLimit", "51"), "invalid");
  assert.equal(parse("searchLimit", "8.5"), "invalid");
  assert.equal(parse("searchLimit", "eight"), "invalid");
  assert.equal(parse("shutdownTimeoutMs", "99"), "invalid");
  assert.equal(parse("requestTimeoutMs", "120001"), "invalid");
});

test("booleans round-trip through their control text", () => {
  assert.deepEqual(parse("autoMemoryEnabled", "true"), {
    kind: "set",
    value: true,
  });
  assert.deepEqual(parse("rootAgentsOnly", "false"), {
    kind: "set",
    value: false,
  });
  assert.equal(spec("autoDreamEnabled").format(true), "true");
  assert.equal(spec("autoDreamEnabled").format(false), "false");
});

test("the endpoint keeps its URL shape and rejects credentials", () => {
  assert.deepEqual(parse("endpoint", " http://127.0.0.1:2333/ "), {
    kind: "set",
    value: "http://127.0.0.1:2333",
  });
  for (const bad of [
    "not-a-url",
    "http://alice:SECRET@localhost:2333",
    "http://localhost:2333?token=SECRET",
    "http://localhost:2333#search",
    "   ",
  ])
    assert.equal(parse("endpoint", bad), "invalid", bad);
});

test("the daily time control writes a cron expression", () => {
  assert.deepEqual(parse("dreamCron", "09:30"), {
    kind: "set",
    value: "30 9 * * *",
  });
  assert.deepEqual(parse("dreamCron", "0:05"), {
    kind: "set",
    value: "5 0 * * *",
  });
  assert.equal(spec("dreamCron").format("30 9 * * *"), "09:30");
  assert.equal(spec("dreamCron").format("*/5 * * * *"), "");
  assert.equal(parse("dreamCron", "24:00"), "invalid");
  assert.equal(parse("dreamCron", "09:60"), "invalid");
  assert.equal(parse("dreamCron", ""), "invalid");
  assert.equal(timeFromCron("5 0 * * *"), "00:05");
});

test("language and timezone only accept what the Host accepts", () => {
  assert.deepEqual(parse("language", "en"), { kind: "set", value: "en" });
  assert.deepEqual(parse("language", "zh"), { kind: "set", value: "zh" });
  assert.equal(parse("language", "fr"), "invalid");
  assert.equal(spec("language").format(undefined), "zh");
  assert.deepEqual(parse("timezone", " Asia/Shanghai "), {
    kind: "set",
    value: "Asia/Shanghai",
  });
  assert.equal(parse("timezone", "Mars/Olympus"), "invalid");
  assert.equal(parse("timezone", ""), "invalid");
  assert.equal(validTimezone("Europe/Berlin"), true);
});

test("the guidance text is taken verbatim", () => {
  assert.deepEqual(parse("dreamHint", "summarise the week"), {
    kind: "set",
    value: "summarise the week",
  });
  assert.deepEqual(parse("dreamHint", ""), { kind: "set", value: "" });
});
