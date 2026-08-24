import assert from "node:assert/strict";
import test from "node:test";
import { getLanguage } from "../app/files-workspace/get-language.ts";
import { parseMarkdownFrontmatter } from "../app/files-workspace/markdown.ts";
import { clampNavigatorWidth } from "../app/files-workspace/panel-resize.ts";
import { buildTree } from "../app/workspace-files.ts";

test("Monaco language mapping stays compatible with QwenPaw", () => {
  assert.equal(getLanguage("src/page.tsx"), "typescript");
  assert.equal(getLanguage("digest/topic.md"), "markdown");
  assert.equal(getLanguage("notes/README"), "plaintext");
});

test("Markdown preview separates simple frontmatter like QwenPaw", () => {
  assert.deepEqual(
    parseMarkdownFrontmatter(
      "---\nname: topic\ndescription: hello\n---\n# Body",
    ),
    {
      body: "# Body",
      entries: [
        { key: "name", value: "topic" },
        { key: "description", value: "hello" },
      ],
    },
  );
});

test("Navigator resizing preserves usable widths for both panes", () => {
  assert.equal(clampNavigatorWidth(100, 1200), 220);
  assert.equal(clampNavigatorWidth(360, 1200), 360);
  assert.equal(clampNavigatorWidth(1000, 1200), 780);
});

test("File tree preserves newest-modified-first ordering from the workspace API", () => {
  const tree = buildTree(
    [
      "daily/older/note.md",
      "digest/recent.md",
      "daily/older/earliest.md",
      "root-old.md",
    ],
    new Set(["md"]),
  );

  assert.deepEqual(
    tree.map((node) => node.path),
    ["daily", "digest", "root-old.md"],
  );
  assert.deepEqual(
    tree[0].children[0].children.map((node) => node.name),
    ["note.md", "earliest.md"],
  );
});
