import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import {
  createDemoWorkspace,
  memoryTags,
  wikilinks,
  DEMO_STORAGE_KEY,
} from "../demo/workspace.ts";
import { scenarios } from "../demo/scenarios.ts";

function storage() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}
const note = (tags, body = "Offline release") =>
  `---\nname: Example\nmemory_tags: ${tags}\n---\n${body}`;
const seeds = {
  "digest/wiki/a.md": note(
    '["aurora", "local_first"]',
    "Offline plan [[digest/wiki/b.md#Scope|Release]]",
  ),
  "digest/wiki/b.md": note('["release"]'),
};

function content(language) {
  const directory = new URL(`../demo/content/${language}/`, import.meta.url);
  return Object.fromEntries(
    fs
      .readdirSync(directory, { recursive: true })
      .filter((file) => file.endsWith(".md"))
      .map((file) => [
        file.split(path.sep).join("/"),
        fs.readFileSync(new URL(file, directory), "utf8"),
      ]),
  );
}

test("memory_tags respects strict lists, normalization, deduplication, and the default three-tag limit", () => {
  assert.deepEqual(
    memoryTags(
      note(
        '[true, 1.5, {}, [], " Local First ", "LOCAL FIRST", "中文", 42, "extra"]',
      ),
    ),
    ["local_first", "中文", "42"],
  );
  assert.deepEqual(memoryTags(note("[aurora, 'Release', personal]")), [
    "aurora",
    "release",
    "personal",
  ]);
  assert.deepEqual(memoryTags(note('["tag,with,commas", "release"]')), [
    "tag,with,commas",
    "release",
  ]);
  assert.deepEqual(
    memoryTags(
      note(`\n  - Local First\n  - RELEASE\n  - local first\n  - personal`),
    ),
    ["local_first", "release", "personal"],
  );
  assert.deepEqual(memoryTags(note("[aurora, release] # shared topic")), [
    "aurora",
    "release",
  ]);
  assert.deepEqual(memoryTags(note("[broken")), []);
  assert.deepEqual(memoryTags(note("aurora")), []);
  assert.deepEqual(memoryTags(note(`["${"a".repeat(65)}", "---", false]`)), []);
  assert.deepEqual(memoryTags("# Body\nmemory_tags: [aurora]"), []);
});

test("saving updates tags, counts, search and graph from files and survives a reload", () => {
  const saved = storage();
  const workspace = createDemoWorkspace(seeds, saved);
  const initial = workspace.read("digest/wiki/a.md");
  assert.equal(workspace.search("offline", ["local_first"]).length, 1);
  workspace.save(
    "digest/wiki/a.md",
    note('["writing"]', "Updated [[missing.md]]"),
    initial.stat.mtime,
  );
  const restored = createDemoWorkspace(seeds, saved);
  assert.equal(restored.search("offline", ["local_first"]).length, 0);
  assert.equal(
    restored.search("updated", ["writing"])[0].path,
    "digest/wiki/a.md",
  );
  assert.deepEqual(restored.tags(), [
    { tag: "release", count: 1 },
    { tag: "writing", count: 1 },
  ]);
  assert.ok(
    restored.graph().edges.some((edge) => edge.target === "missing.md"),
  );
  assert.equal(
    restored.graph().nodes.find((node) => node.id === "missing.md").indexed,
    false,
  );
  assert.ok(
    !restored
      .graph()
      .edges.some(
        (edge) =>
          edge.source === "digest/wiki/a.md" &&
          edge.target === "digest/wiki/b.md",
      ),
  );
  assert.match(seeds["digest/wiki/a.md"], /local_first/);
  restored.reset();
  assert.equal(restored.read("digest/wiki/a.md").content, initial.content);
});

test("tag filters match any selected tag, while keywords constrain every result", () => {
  const workspace = createDemoWorkspace(seeds);
  assert.equal(workspace.search("", ["local_first", "release"]).length, 2);
  assert.equal(
    workspace.search("offline plan", ["local_first", "release"]).length,
    1,
  );
  assert.equal(workspace.search("", ["nonexistent"]).length, 0);
  const normalized = createDemoWorkspace({
    "note.md": note('["Local First"]'),
  });
  assert.equal(normalized.search("local_first", ["local_first"]).length, 1);
});

test("stale browser writes and unknown paths are rejected without losing the newer file", () => {
  const saved = storage();
  const one = createDemoWorkspace(seeds, saved);
  const two = createDemoWorkspace(seeds, saved);
  const initial = one.read("digest/wiki/a.md");
  two.save("digest/wiki/a.md", "Newer content", initial.stat.mtime);
  assert.throws(
    () => one.save("digest/wiki/a.md", "Stale content", initial.stat.mtime),
    /changed/,
  );
  assert.equal(one.read("digest/wiki/a.md").content, "Newer content");
  for (const file of ["../escape.md", "toString", "__proto__", "new.md"]) {
    assert.throws(() => one.read(file), /not found/);
    assert.throws(() => one.save(file, "content"), /not found/);
  }
});

test("failed persistence does not report success or replace the source", () => {
  const workspace = createDemoWorkspace(seeds, {
    getItem: () => null,
    setItem: () => {
      throw new Error("Quota exceeded");
    },
    removeItem: () => {},
  });
  assert.throws(() => workspace.save("digest/wiki/a.md", "not saved"), /Quota/);
  assert.equal(
    workspace.read("digest/wiki/a.md").content,
    seeds["digest/wiki/a.md"],
  );
});

test("corrupt persisted content falls back to examples, and languages stay isolated", () => {
  const saved = storage();
  saved.setItem(DEMO_STORAGE_KEY, "broken JSON");
  assert.equal(
    createDemoWorkspace(seeds, saved).read("digest/wiki/a.md").content,
    seeds["digest/wiki/a.md"],
  );
  saved.setItem(
    DEMO_STORAGE_KEY,
    JSON.stringify({
      "digest/wiki/a.md": { content: "bad", mtime: "invalid" },
    }),
  );
  assert.equal(
    createDemoWorkspace(seeds, saved).read("digest/wiki/a.md").content,
    seeds["digest/wiki/a.md"],
  );
  createDemoWorkspace(seeds, saved, "zh").save("digest/wiki/a.md", "中文修改");
  assert.notEqual(
    createDemoWorkspace(seeds, saved, "en").read("digest/wiki/a.md").content,
    "中文修改",
  );
});

for (const language of ["zh", "en"])
  test(`${language} examples form a complete source-to-memory chain with valid tags and scripted citations`, () => {
    const files = content(language);
    assert.equal(Object.keys(files).length, 21);
    assert.deepEqual(
      Object.keys(files).sort(),
      Object.keys(content(language === "zh" ? "en" : "zh")).sort(),
    );
    for (const [file, text] of Object.entries(files)) {
      assert.ok(text.length > 150, `${file}: complete content`);
      assert.equal(memoryTags(text).length, 3, `${file}: three valid tags`);
      for (const { target } of wikilinks(text))
        assert.ok(
          Object.hasOwn(files, target),
          `${file}: missing target ${target}`,
        );
    }
    for (const scenario of scenarios[language]) {
      assert.ok(createDemoWorkspace(files).search("", [scenario.tag]).length);
      for (const file of scenario.paths) assert.ok(Object.hasOwn(files, file));
      for (const { target } of wikilinks(scenario.answer))
        assert.ok(Object.hasOwn(files, target));
    }
    const graph = createDemoWorkspace(files).graph();
    for (const category of ["wiki", "personal", "procedure"])
      assert.ok(
        graph.edges.some((edge) => edge.source === `virtual:${category}`),
      );
  });
