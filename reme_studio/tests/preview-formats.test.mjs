import assert from "node:assert/strict";
import test from "node:test";
import { parseCsv } from "../app/files-workspace/csv.ts";
import { parseMarkdownFrontmatter } from "../app/files-workspace/markdown.ts";
import { browserStorage } from "../app/browser-storage.ts";

test("CSV records preserve multiline cells, escaped quotes and trailing whitespace", () => {
  assert.deepEqual(
    parseCsv(
      '\uFEFFname,detail,note\r\nAlice,"first\r\nsecond","a ""quote"""\r\nBob,last,  \r\n',
    ),
    [
      ["name", "detail", "note"],
      ["Alice", "first\r\nsecond", 'a "quote"'],
      ["Bob", "last", "  "],
    ],
  );
  assert.deepEqual(parseCsv('one,two,\n"",x,'), [
    ["one", "two", ""],
    ["", "x", ""],
  ]);
  assert.deepEqual(parseCsv(""), []);
});

test("Markdown properties retain structured YAML and block strings", () => {
  const result = parseMarkdownFrontmatter(
    '\uFEFF---\r\ntitle: "Research: notes"\r\ndescription: |\r\n  first line\r\n  second line\r\nmemory_tags:\r\n  - agents\r\n  - memory\r\nmetadata:\r\n  source: local\r\n---\r\n# Body',
  );
  assert.equal(result.body, "# Body");
  assert.deepEqual(result.entries, [
    { key: "title", value: "Research: notes" },
    { key: "description", value: "first line\nsecond line\n" },
    { key: "memory_tags", value: "- agents\n- memory" },
    { key: "metadata", value: "source: local" },
  ]);
});

test("invalid frontmatter and non-leading fences remain visible without losing source", () => {
  for (const text of [
    "---\nname: [unterminated\n---\nBody",
    "---\n- scalar\n---\nBody",
    "# Intro\n---\nname: value\n---\nBody",
  ]) {
    assert.deepEqual(parseMarkdownFrontmatter(text), {
      body: text,
      entries: [],
    });
  }
});

test("blocked browser storage falls back to the current in-memory UI", () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() {
      throw new Error("Storage is blocked");
    },
  });
  try {
    assert.equal(browserStorage.getItem("theme"), null);
    assert.doesNotThrow(() => browserStorage.setItem("theme", "dark"));
    assert.doesNotThrow(() => browserStorage.removeItem("theme"));
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else Reflect.deleteProperty(globalThis, "localStorage");
  }
});
