import assert from "node:assert/strict";
import test from "node:test";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const docs = join(dirname(fileURLToPath(import.meta.url)), "..", "docs");

/**
 * Whether the DSH Web GUI renders this code span as a clickable link.
 *
 * The GUI's markdown renderer has an explicit case for it: an `inlineCode`
 * node whose content is exactly an http(s) URL becomes
 * `<code><a href="...">URL</a></code>` — the code chip is a link.
 *
 *   function uU(e){ if(e.trim()===e) try{ const t=new URL(e).protocol;
 *       return t==="http:"||t==="https:"?e:void 0 }catch{return} }
 *
 * (from `@deepseek-ai/dsh-web-frontend`, the `inlineCode` case of the
 * markdown renderer.) A standard CommonMark renderer leaves a code span inert,
 * so this only shows up in the GUI — and an example endpoint is not something a
 * reader should be able to click.
 */
function guiLinkifies(code) {
  if (code.trim() !== code) return undefined;
  try {
    const protocol = new URL(code).protocol;
    return protocol === "http:" || protocol === "https:" ? code : undefined;
  } catch {
    return undefined;
  }
}

/** Code spans of a markdown document, fenced blocks left out. */
function codeSpans(text) {
  const withoutFences = text.replace(/^```[\s\S]*?^```/gm, "");
  return [...withoutFences.matchAll(/`([^`\n]+)`/g)].map((match) => match[1]);
}

test("the user docs hold no code span the GUI turns into a link", (t) => {
  for (const file of ["environment.md", "environment_ZH.md"]) {
    const path = join(docs, file);
    // The packed plugin does not ship the docs; there is nothing to check
    // there.
    if (!existsSync(path)) {
      t.skip(`${file} is not part of the package`);
      continue;
    }
    const offenders = codeSpans(readFileSync(path, "utf8")).filter(
      (code) => guiLinkifies(code) !== undefined,
    );
    assert.deepEqual(
      offenders,
      [],
      `${file} holds a code span the GUI renders as a clickable link`,
    );
  }
});
