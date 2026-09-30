import assert from "node:assert/strict";
import test from "node:test";
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ReactMarkdown from "react-markdown";
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

function parse(content, wikilinks = true) {
  const processor = unified().use(remarkParse).use(remarkGfm);
  if (wikilinks) processor.use(remarkWikilinks);
  return processor.runSync(processor.parse(content));
}

function render(content, wikilinks = true) {
  return renderToStaticMarkup(
    createElement(
      ReactMarkdown,
      { remarkPlugins: wikilinks ? [remarkGfm, remarkWikilinks] : [remarkGfm] },
      content,
    ),
  );
}

function links(tree) {
  return (
    tree.children?.flatMap((node) =>
      node.type === "link" ? [node] : links(node),
    ) || []
  );
}

test("wikilinks open literal workspace targets and preserve aliases", () => {
  const result = links(
    parse("See [[digest/a.md#Section|Decision]] and [[b]] today."),
  );
  assert.deepEqual(
    result.map((node) => node.url),
    ["#reme-file=digest%2Fa.md", "#reme-file=b"],
  );
  assert.equal(result[0].children[0].value, "Decision");
});

test("Markdown punctuation in targets and aliases stays literal", () => {
  for (const [target, alias] of [
    ["resources/release~draft~.md", "**发布状态**"],
    ["notes/a*b*.md", "a `code` label"],
    ["notes/a&copy;.md", "https://example.com"],
    ["notes/a\\b.md", "~~old~~"],
  ]) {
    const result = links(parse(`Before [[${target}#Scope|${alias}]] after.`));
    assert.equal(result.length, 1);
    assert.equal(workspaceLinkTarget(result[0].url), target);
    assert.deepEqual(result[0].children, [{ type: "text", value: alias }]);
  }
});

test("code examples and existing Markdown links are not rewritten", () => {
  const content = [
    "```md",
    "[[sample.md]]",
    "```",
    "",
    "`[[sample.md]]` and [See [[sample.md]]](https://example.com)",
    "![See [[sample.md]]](image.png)",
  ].join("\n");
  assert.equal(render(content), render(content, false));
});

test("malformed wikilinks leave normal Markdown parsing unchanged", () => {
  for (const content of [
    "[[ ]]",
    "[[a.md",
    "[[a.md] text",
    "[[a.md\nb.md]]",
    "[[a.md#]]",
    "[[a.md|]]",
    "[ordinary](https://example.com)",
  ])
    assert.deepEqual(parse(content), parse(content, false));
});

test("wikilinks work beside ordinary emphasis and consecutive links", () => {
  const tree = parse("**Before [[a.md|first]]** [[b.md]][[c.md]] *after*");
  assert.deepEqual(
    links(tree).map((node) => workspaceLinkTarget(node.url)),
    ["a.md", "b.md", "c.md"],
  );
  assert.equal(tree.children[0].children[0].type, "strong");
  assert.equal(tree.children[0].children.at(-1).type, "emphasis");
});

test("reference links and images preserve literal wikilinks in their labels", () => {
  for (const content of [
    "[See [[sample.md]]][ref]\n\n[ref]: https://example.com",
    "[See [[sample.md]]][REF]\n\n[ref]: https://example.com",
    "![See [[sample.md]]][ref]\n\n[ref]: image.png",
  ]) {
    assert.equal(render(content), render(content, false));
    assert.doesNotMatch(render(content), /#reme-file=/);
  }
});

test("escaped brackets and character entities remain literal text", () => {
  for (const content of [
    String.raw`\[\[example.md\]\]`,
    "&#91;&#91;example.md&#93;&#93;",
    "&#x5b;&#x5b;example.md&#x5d;&#x5d;",
    String.raw`\[[example.md]]`,
  ]) {
    assert.equal(render(content), render(content, false));
    assert.equal(links(parse(content)).length, 0);
  }
  const mixed = String.raw`\[\[literal.md\]\] [[real.md]] &#91;&#91;entity.md&#93;&#93;`;
  assert.deepEqual(
    links(parse(mixed)).map((node) => node.url),
    ["#reme-file=real.md"],
  );
  assert.match(render(mixed), /\[\[literal\.md\]\]/);
  assert.match(render(mixed), /\[\[entity\.md\]\]/);
});
