import assert from "node:assert/strict";
import test from "node:test";

import { parseMarkdownFrontmatter } from "../dist/dsh/client/frontmatter.js";

test("separates top-level frontmatter from Markdown content", () => {
  assert.deepEqual(
    parseMarkdownFrontmatter(
      "---\nname: oauth2\ndescription: delegated authorization\n  framework\n---\n# OAuth 2.0",
    ),
    {
      body: "# OAuth 2.0",
      entries: [
        { key: "name", value: "oauth2" },
        { key: "description", value: "delegated authorization\nframework" },
      ],
    },
  );
});

test("keeps ordinary Markdown unchanged", () => {
  assert.deepEqual(parseMarkdownFrontmatter("# Journal\n\nRemember this."), {
    body: "# Journal\n\nRemember this.",
    entries: [],
  });
});
