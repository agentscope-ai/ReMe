import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";
import {
  animationLabels,
  defaultLanguage,
  deserializeError,
  errorText,
  messages,
  serializeError,
  translate,
  UiError,
} from "../src/shared/i18n.ts";

test("paired translations have matching placeholders and interpolation preserves literal user data", async () => {
  assert.equal(defaultLanguage, "zh");
  for (const [key, [zh, en]] of Object.entries(messages)) {
    assert.ok(zh && en, key);
    assert.deepEqual(
      zh.match(/\{\w+\}/g)?.sort(),
      en.match(/\{\w+\}/g)?.sort(),
      key,
    );
  }
  assert.equal(
    translate("en", "workspace", { path: "$&/{example}" }),
    "Workspace: $&/{example}",
  );
  const manifest = JSON.parse(
    await readFile(
      new URL(
        "../../assets/remy/anime/animations/v1/manifest.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
  assert.deepEqual(
    manifest.animations.map((clip: { id: string }) => clip.id).sort(),
    Object.keys(animationLabels).sort(),
  );
  for (const clip of manifest.animations)
    assert.equal(
      "label" in clip,
      false,
      "runtime labels belong to the language catalog",
    );
});

test("localized errors survive plain-data IPC and external service messages stay unchanged", () => {
  const error = deserializeError(
    structuredClone(
      serializeError(new UiError("missingJob", { job: "pa_events" })),
    ),
  );
  assert.equal(
    errorText(error, "en"),
    "The service has not enabled pa_events. Start it with reme start config=proactive",
  );
  assert.match(errorText(error, "zh"), /服务未启用 pa_events/);
  assert.equal(
    errorText(
      deserializeError(serializeError(new Error("模型原始错误"))),
      "en",
    ),
    "模型原始错误",
  );
});

test("desktop source keeps user-facing Chinese literals in the shared catalog", async () => {
  const visit = async (directory: URL): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const file = new URL(
        entry.name + (entry.isDirectory() ? "/" : ""),
        directory,
      );
      if (entry.isDirectory()) {
        await visit(file);
        continue;
      }
      if (!/\.tsx?$/.test(entry.name) || entry.name === "i18n.ts") continue;
      const source = ts.createSourceFile(
        file.pathname,
        await readFile(file, "utf8"),
        ts.ScriptTarget.Latest,
        true,
      );
      const inspect = (node: ts.Node): void => {
        if (
          ts.isStringLiteralLike(node) ||
          ts.isJsxText(node) ||
          ts.isTemplateHead(node) ||
          ts.isTemplateMiddle(node) ||
          ts.isTemplateTail(node)
        )
          assert.ok(
            !/\p{Script=Han}/u.test(node.text),
            `${file.pathname}: ${node.text}`,
          );
        ts.forEachChild(node, inspect);
      };
      inspect(source);
    }
  };
  await visit(new URL("../src/", import.meta.url));
});
