import assert from "node:assert/strict";
import test from "node:test";
import {
  absoluteWorkspacePath,
  appendWorkspaceFileReference,
} from "../app/workspace-drag.ts";

test("dragged workspace files use the ReMe service absolute path", () => {
  assert.equal(
    absoluteWorkspacePath("/Users/yuli/.reme/", "daily/2026-08-05.md"),
    "/Users/yuli/.reme/daily/2026-08-05.md",
  );
  assert.equal(
    absoluteWorkspacePath("C:\\ReMe\\workspace\\", "daily/entry.md"),
    "C:\\ReMe\\workspace\\daily\\entry.md",
  );
});

test("dropping a file appends one delimited reference without duplicates", () => {
  const path = "/Users/yuli/My Memory/daily/today.md";
  const first = appendWorkspaceFileReference("请总结", path);

  assert.equal(first, "请总结\n`/Users/yuli/My Memory/daily/today.md`");
  assert.equal(appendWorkspaceFileReference(first, path), first);
});
