import assert from "node:assert/strict";
import test from "node:test";
import { workspacePlacement } from "../src/main/placement.ts";
import { requestWorkspace } from "../src/main/workspace.ts";
import { ReMeClient } from "../src/main/reme-client.ts";
import {
  buildTree,
  documentPresentation,
  parseMarkdown,
  resolveDocumentLink,
} from "../src/renderer/knowledge/model.ts";
import type { WorkspaceRequest } from "../src/shared/protocol.ts";

test("workspace follows display proportions and recovers bounds from disconnected monitors", () => {
  const area = { x: -1600, y: 25, width: 1600, height: 950 };
  const display = { ...area, height: 1000 };
  const bounds = workspacePlacement(area, display);
  assert.ok(Math.abs(bounds.width / bounds.height - 1.6) < 0.003);
  const saved = { x: 3000, y: 4000, width: 1800, height: 1100 };
  assert.deepEqual(workspacePlacement(area, display, saved), {
    x: -1600,
    y: 25,
    width: 1600,
    height: 950,
  });
});

test("multiline YAML tags and wikilinks stay structured, safe and directory-aware", () => {
  const markdown =
    "---\nname: Memory\nmemory_tags:\n  - ReMe\n  - 'C++'\ndescription: |\n  A connected memory.\n---\n# Body";
  const parsed = parseMarkdown(markdown);
  assert.deepEqual(parsed.tags, ["ReMe", "C++"]);
  assert.equal(parsed.body, "# Body");
  assert.equal(parsed.metadata.description, "A connected memory.\n");
  const paths = ["digest/wiki/a.md", "daily/b.md"];
  assert.equal(
    resolveDocumentLink("../../daily/b.md", paths[0], paths),
    "daily/b.md",
  );
  assert.equal(resolveDocumentLink("a", paths[1], paths), paths[0]);
  assert.equal(
    resolveDocumentLink("../../../escape.md", paths[0], paths),
    undefined,
  );
  assert.equal(
    resolveDocumentLink("file:///etc/passwd", paths[0], paths),
    undefined,
  );
  assert.equal(
    buildTree(paths.slice(0, 1), "digest")[0].children?.[0].path,
    paths[0],
  );
});

test("frontmatter preserves all properties and promotes a body title without a duplicate filename", () => {
  const parsed = parseMarkdown(
    "\uFEFF---\r\ncreated_at: '2026-11-08T10:53:47+08:00'\r\nstatus: unsent\r\ntarget_session_id: session-1\r\nmemory_tags: [ReMe, 2026]\r\noptions:\r\n  enabled: true\r\n---\r\n\r\n# A **real** document title\r\n\r\n## Section\r\nText.",
  );
  assert.equal(parsed.frontmatterError, false);
  assert.equal(parsed.metadata.status, "unsent");
  assert.equal(parsed.metadata.target_session_id, "session-1");
  assert.deepEqual(parsed.metadata.options, { enabled: true });
  assert.deepEqual(parsed.tags, ["ReMe", "2026"]);
  const display = documentPresentation("daily/interest_105347.md", parsed);
  assert.equal(display.title, "A real document title");
  assert.equal(display.body, "## Section\r\nText.");
});

test("document links keep relative Markdown paths separate from workspace wikilinks", () => {
  const paths = ["note.md", "digest/wiki/known.md", "daily/a.md"];
  assert.equal(
    resolveDocumentLink("note.md", "daily/a.md", paths),
    "daily/note.md",
  );
  assert.equal(
    resolveDocumentLink("digest/wiki/new.md", "", paths),
    "digest/wiki/new.md",
  );
  assert.equal(
    resolveDocumentLink("digest/wiki/new", "", paths),
    "digest/wiki/new.md",
  );
  assert.equal(resolveDocumentLink("known", "", paths), "digest/wiki/known.md");
  assert.equal(resolveDocumentLink("../escape.md", "", paths), undefined);
});

test("heading selection respects empty metadata, code fences, Setext and explicit titles", () => {
  const empty = parseMarkdown("---\n---\n# Heading\n\nBody");
  assert.equal(empty.body, "# Heading\n\nBody");
  assert.equal(documentPresentation("note.md", empty).title, "Heading");
  assert.equal(
    documentPresentation("note.md", parseMarkdown("Setext title\n===\n\nBody"))
      .body,
    "Body",
  );
  const fenced = parseMarkdown("```md\n# Code, not a title\n```\n");
  assert.equal(documentPresentation("note.md", fenced).title, "note");
  assert.equal(documentPresentation("note.md", fenced).body, fenced.body);
  const same = parseMarkdown("---\ntitle: Heading\n---\n# Heading\n\nBody");
  assert.deepEqual(documentPresentation("note.md", same), {
    title: "Heading",
    body: "Body",
  });
  const different = parseMarkdown(
    "---\nname: Notebook\n---\n# Section\n\nBody",
  );
  assert.equal(documentPresentation("note.md", different).body, different.body);
  assert.equal(
    documentPresentation(
      "note.md",
      parseMarkdown("# Title\n\n    indented code\n"),
    ).body,
    "    indented code\n",
  );
});

test("invalid frontmatter remains inspectable without turning YAML into Markdown headings", () => {
  for (const header of ["tags: [unclosed", "[not, a, mapping]"]) {
    const parsed = parseMarkdown(`---\n${header}\n---\n# Body\n\nText`);
    assert.equal(parsed.frontmatterError, true);
    assert.equal(parsed.rawFrontmatter.trim(), header);
    assert.equal(documentPresentation("note.md", parsed).title, "Body");
  }
});

test("workspace IPC cannot invoke mutation jobs or escape workspace-relative paths", async () => {
  const client = new ReMeClient("http://unused");
  for (const request of [
    { type: "load", path: "../secrets" },
    { type: "load", path: "/etc/passwd" },
    { type: "save", path: "note.md" },
    { type: "tags", page: 0 },
    { type: "search", path: "", query: "", tags: [] },
  ])
    await assert.rejects(() =>
      requestWorkspace(
        client,
        request as WorkspaceRequest,
        new AbortController().signal,
      ),
    );
});

test("full responses preserve file-list and search metadata", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        success: true,
        answer: "human-readable",
        metadata: { items: ["daily/a.md"] },
      }),
    );
  try {
    const result = await requestWorkspace(
      new ReMeClient("http://unused"),
      { type: "list", path: "daily" },
      new AbortController().signal,
    );
    assert.deepEqual(result.metadata.items, ["daily/a.md"]);
  } finally {
    globalThis.fetch = original;
  }
});

test("workspace discovery follows the configured tag index and excludes private configuration", async () => {
  const client = new ReMeClient("http://unused");
  client.call = async <T>() =>
    ({
      workspace_dir: "/workspace",
      daily_dir: "notes",
      digest_dir: "knowledge",
      jobs: {
        load: { backend: "base" },
        internal: { backend: "base", enable_serve: false },
      },
      components: {
        file_store: { default: { tag_index: "custom" } },
        tag_index: { custom: { tag_key: "entities" } },
      },
      secret: "must not reach renderer",
    }) as T;
  const result = await requestWorkspace(
    client,
    { type: "config" },
    new AbortController().signal,
  );
  assert.deepEqual(result.answer, {
    workspace_dir: "/workspace",
    daily_dir: "notes",
    digest_dir: "knowledge",
    tag_key: "entities",
    jobs: ["load"],
  });
});
