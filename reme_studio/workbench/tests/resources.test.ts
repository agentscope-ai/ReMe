import assert from "node:assert/strict";
import test from "node:test";
import { plainText } from "../src/resources.ts";

test("resource summaries keep text without Markdown control characters", () => {
  assert.equal(
    plainText("## 新发现\n\n**实验完成**，查看[原文](daily/result.md)。"),
    "新发现 实验完成，查看原文。",
  );
  assert.equal(plainText("- A\n- B\n\n`code`"), "A B code");
});
