import assert from "node:assert/strict";
import test from "node:test";
import { translate } from "../app/i18n.ts";

test("translations support Chinese, English, and interpolation", () => {
  assert.equal(
    translate("zh", "openChat", { mode: "Proactive Chat" }),
    "打开 Proactive Chat",
  );
  assert.equal(translate("en", "openChat", { mode: "Chat" }), "Open Chat");
  assert.equal(
    translate("en", "openingFile", { path: "daily/today.md" }),
    "Opening daily/today.md",
  );
});
