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
  const render = (plugins) =>
    renderToStaticMarkup(
      createElement(
        ReactMarkdown,
        {
          remarkPlugins: plugins,
        },
        content,
      ),
    );
  assert.equal(render([remarkGfm, remarkWikilinks]), render([remarkGfm]));
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
