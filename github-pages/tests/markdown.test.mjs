import assert from "node:assert/strict";
import test from "node:test";
import { stripMarkdownFrontmatter } from "../src/markdown.js";

test("strips leading YAML frontmatter before rendering", () => {
  assert.equal(
    stripMarkdownFrontmatter("---\nname: reme_memory\ndescription: Memory skill\n---\n\n# ReMe Memory\n"),
    "\n# ReMe Memory\n",
  );
});

test("preserves Markdown without frontmatter", () => {
  const markdown = "# ReMe Memory\n\nContent\n";
  assert.equal(stripMarkdownFrontmatter(markdown), markdown);
});
