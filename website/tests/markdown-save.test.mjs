import assert from "node:assert/strict";
import test from "node:test";
import { markMarkdownContentSaved } from "../app/markdown-save.ts";

test("saving an older submission preserves edits made while the request was pending", () => {
  const tab = {
    id: "file:note.md",
    type: "markdown",
    title: "note.md",
    path: "note.md",
    content: "edited while saving",
    savedContent: "before save",
    mtime: "old-mtime",
  };

  const saved = markMarkdownContentSaved(tab, "submitted content", "new-mtime");

  assert.equal(saved.content, "edited while saving");
  assert.equal(saved.savedContent, "submitted content");
  assert.equal(saved.mtime, "new-mtime");
  assert.notEqual(saved.content, saved.savedContent);
});
