import assert from "node:assert/strict";
import test from "node:test";
import {
  edgePath,
  graphBelowRoot,
  layoutGraph,
  reciprocalEdgeKeys,
} from "../app/files-workspace/memory-graph.ts";

const snapshot = {
  version: 1,
  nodes: [
    {
      id: "virtual:wiki",
      path: "digest/wiki",
      name: "wiki",
      description: "",
      indexed: false,
      virtual: true,
    },
    {
      id: "virtual:personal",
      path: "digest/personal",
      name: "personal",
      description: "",
      indexed: false,
      virtual: true,
    },
    {
      id: "digest/wiki/a.md",
      path: "digest/wiki/a.md",
      name: "A",
      description: "",
      indexed: true,
      virtual: false,
    },
    {
      id: "digest/personal/b.md",
      path: "digest/personal/b.md",
      name: "B",
      description: "",
      indexed: true,
      virtual: false,
    },
    {
      id: "daily/note.md",
      path: "daily/note.md",
      name: "Note",
      description: "",
      indexed: true,
      virtual: false,
    },
    {
      id: "virtual:procedure",
      path: "digest/procedure",
      name: "procedure",
      description: "",
      indexed: false,
      virtual: true,
    },
    {
      id: "digest/procedure/c.md",
      path: "digest/procedure/c.md",
      name: "C",
      description: "",
      indexed: true,
      virtual: false,
    },
    {
      id: "daily/unrelated.md",
      path: "daily/unrelated.md",
      name: "Unrelated",
      description: "",
      indexed: true,
      virtual: false,
    },
  ],
  edges: [
    { source: "virtual:wiki", target: "digest/wiki/a.md", target_anchor: null },
    {
      source: "virtual:personal",
      target: "digest/personal/b.md",
      target_anchor: null,
    },
    {
      source: "digest/wiki/a.md",
      target: "daily/note.md",
      target_anchor: null,
    },
    {
      source: "digest/wiki/a.md",
      target: "digest/personal/b.md",
      target_anchor: null,
    },
    {
      source: "digest/personal/b.md",
      target: "digest/wiki/a.md",
      target_anchor: null,
    },
    {
      source: "virtual:procedure",
      target: "digest/procedure/c.md",
      target_anchor: null,
    },
    {
      source: "digest/procedure/c.md",
      target: "daily/unrelated.md",
      target_anchor: null,
    },
  ],
};

test("memory graph keeps only nodes reachable below the selected category", () => {
  const graph = graphBelowRoot(snapshot, "wiki");

  assert.deepEqual(
    graph.nodes.map((node) => node.id),
    [
      "virtual:wiki",
      "digest/wiki/a.md",
      "digest/personal/b.md",
      "daily/note.md",
    ],
  );
  assert.equal(graph.edges.length, 4);
});

test("memory graph excludes daily nodes below another category", () => {
  const graph = graphBelowRoot(snapshot, "wiki");

  assert.ok(!graph.nodes.some((node) => node.id === "daily/unrelated.md"));
  assert.ok(!graph.nodes.some((node) => node.id === "digest/procedure/c.md"));
});

test("memory graph uses stable radial layers and curves reciprocal links", () => {
  const graph = graphBelowRoot(snapshot, "wiki");
  const positioned = layoutGraph(graph);
  const root = positioned.byId.get("virtual:wiki");
  const direct = positioned.byId.get("digest/wiki/a.md");
  const leaf = positioned.byId.get("daily/note.md");

  assert.deepEqual(
    { x: root.x, y: root.y, layer: root.layer },
    { x: 540, y: 340, layer: 0 },
  );
  assert.equal(direct.layer, 1);
  assert.equal(leaf.layer, 2);

  const reciprocal = reciprocalEdgeKeys(graph.edges);
  const edge = graph.edges.find(
    (item) =>
      item.source === "digest/wiki/a.md" &&
      item.target === "digest/personal/b.md",
  );
  assert.match(edgePath(edge, positioned.byId, reciprocal), / Q /);
});
