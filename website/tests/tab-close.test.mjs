import assert from "node:assert/strict";
import test from "node:test";
import { hasUnsavedChanges, unsavedTabsClosedBy } from "../app/tab-close.ts";

const saved = {
  id: "file:saved.md",
  type: "markdown",
  title: "saved.md",
  path: "saved.md",
  content: "saved",
  savedContent: "saved",
};
const draft = {
  ...saved,
  id: "file:draft.md",
  path: "draft.md",
  content: "unsaved draft",
};
const otherDraft = {
  ...draft,
  id: "file:other.md",
  path: "other.md",
};

test("tab close protection identifies only drafts that would be discarded", () => {
  const tabs = [saved, draft, otherDraft];

  assert.equal(hasUnsavedChanges(saved), false);
  assert.equal(hasUnsavedChanges(draft), true);
  assert.deepEqual(unsavedTabsClosedBy(tabs, draft.id, false), [draft]);
  assert.deepEqual(unsavedTabsClosedBy(tabs, draft.id, true), [otherDraft]);
});
