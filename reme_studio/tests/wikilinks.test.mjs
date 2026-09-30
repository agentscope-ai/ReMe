import assert from "node:assert/strict";
import test from "node:test";
import {
  remarkWikilinks,
  workspaceLinkTarget,
} from "../app/files-workspace/wikilinks.ts";

test("workspace link fragments tolerate malformed Markdown URLs", () => {
  assert.equal(workspaceLinkTarget("#reme-file=digest%2Fa.md"), "digest/a.md");
  assert.equal(workspaceLinkTarget("#reme-file=%E4%B8%AD.md"), "中.md");
  for (const href of [
    undefined,
    "https://example.com",
    "#reme-file=",
    "#reme-file=%",
    "#reme-file=%FF",
  ])
    assert.equal(workspaceLinkTarget(href), undefined);
});

test("wikilinks open literal workspace targets and preserve aliases", () => {
  const tree = {
    type: "root",
    children: [
      {
        type: "paragraph",
        children: [
          {
            type: "text",
            value: "See [[digest/a.md#Section|Decision]] and [[b]] today.",
          },
        ],
      },
    ],
  };
  remarkWikilinks()(tree);
  const links = tree.children[0].children.filter(
    (node) => node.type === "link",
  );
  assert.deepEqual(
    links.map((node) => node.url),
    ["#reme-file=digest%2Fa.md", "#reme-file=b"],
  );
  assert.equal(links[0].children[0].value, "Decision");
});

test("code examples and existing Markdown links are not rewritten", () => {
  const tree = {
    type: "root",
    children: [
      { type: "code", value: "[[sample.md]]" },
      {
        type: "paragraph",
        children: [
          { type: "inlineCode", value: "[[sample.md]]" },
          {
            type: "link",
            url: "https://example.com",
            children: [{ type: "text", value: "[[sample.md]]" }],
          },
        ],
      },
    ],
  };
  const original = structuredClone(tree);
  remarkWikilinks()(tree);
  assert.deepEqual(tree, original);
});
