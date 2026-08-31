import assert from "node:assert/strict";
import test from "node:test";
import { translate } from "../app/i18n.ts";

test("translations support Chinese, English, and interpolation", () => {
  assert.equal(translate("zh", "newAgentChat"), "新建 Agent 对话");
  assert.equal(translate("en", "newAgentChat"), "New Agent chat");
  assert.equal(
    translate("en", "openingFile", { path: "daily/today.md" }),
    "Opening daily/today.md",
  );
});
