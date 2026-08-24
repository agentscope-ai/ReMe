import type {
  GraphSnapshot,
  GraphSnapshotEdge,
  GraphSnapshotNode,
  MemoryGraphRoot,
} from "../types";

export const GRAPH_WIDTH = 1080;
export const GRAPH_HEIGHT = 680;
export const INNER_RING_RADIUS = 164;
export const OUTER_RING_RADIUS = 270;

export interface PositionedGraphNode extends GraphSnapshotNode {
  degree: number;
  layer: 0 | 1 | 2;
  x: number;
  y: number;
  radius: number;
}

export interface PositionedGraph {
  nodes: PositionedGraphNode[];
  byId: Map<string, PositionedGraphNode>;
}

export function nodeLabel(node: GraphSnapshotNode): string {
  return (
    node.name || node.path.split("/").pop()?.replace(/\.md$/i, "") || node.path
  );
}

export function shortNodeLabel(node: GraphSnapshotNode): string {
  const label = nodeLabel(node);
  return label.length > 25 ? `${label.slice(0, 22)}…` : label;
}

export function graphBelowRoot(
  snapshot: GraphSnapshot,
  root: MemoryGraphRoot,
): GraphSnapshot {
  const rootId = `virtual:${root}`;
  if (!snapshot.nodes.some((node) => node.id === rootId))
    return { ...snapshot, nodes: [], edges: [] };
  const outgoing = new Map<string, string[]>();
  snapshot.edges.forEach((edge) =>
    outgoing.set(edge.source, [
      ...(outgoing.get(edge.source) || []),
      edge.target,
    ]),
  );
  const reachable = new Set([rootId]);
  const queue = [rootId];
  while (queue.length) {
    const current = queue.shift() as string;
    (outgoing.get(current) || []).forEach((target) => {
      if (reachable.has(target)) return;
      reachable.add(target);
      queue.push(target);
    });
  }
  return {
    ...snapshot,
    nodes: snapshot.nodes.filter((node) => reachable.has(node.id)),
    edges: snapshot.edges.filter(
      (edge) => reachable.has(edge.source) && reachable.has(edge.target),
    ),
  };
}

function degrees(snapshot: GraphSnapshot): Map<string, number> {
  const result = new Map(snapshot.nodes.map((node) => [node.id, 0]));
  snapshot.edges.forEach((edge) => {
    result.set(edge.source, (result.get(edge.source) || 0) + 1);
    result.set(edge.target, (result.get(edge.target) || 0) + 1);
  });
  return result;
}

export function layoutGraph(snapshot: GraphSnapshot): PositionedGraph {
  const degree = degrees(snapshot);
  const root = snapshot.nodes.find((node) => node.virtual) || snapshot.nodes[0];
  const nodes: PositionedGraphNode[] = snapshot.nodes.map((node) => {
    const nodeDegree = degree.get(node.id) || 0;
    return {
      ...node,
      degree: nodeDegree,
      layer: node.id === root?.id ? 0 : 2,
      x: GRAPH_WIDTH / 2,
      y: GRAPH_HEIGHT / 2,
      radius: node.virtual ? 11 : Math.min(9, 4 + Math.sqrt(nodeDegree) * 1.35),
    };
  });
  const byId = new Map(nodes.map((node) => [node.id, node]));
  if (!root) return { nodes, byId };

  const outgoing = new Map<string, string[]>();
  snapshot.edges.forEach((edge) => {
    if (byId.has(edge.source) && byId.has(edge.target)) {
      outgoing.set(edge.source, [
        ...(outgoing.get(edge.source) || []),
        edge.target,
      ]);
    }
  });
  const inner = [...new Set(outgoing.get(root.id) || [])]
    .filter((id) => id !== root.id && byId.has(id))
    .sort(
      (left, right) =>
        (degree.get(right) || 0) - (degree.get(left) || 0) ||
        left.localeCompare(right),
    );
  const innerSet = new Set(inner);
  const startAngle = -Math.PI / 2;
  const innerAngles = new Map<string, number>();
  inner.forEach((id, index) => {
    const angle =
      startAngle + (index / Math.max(1, inner.length)) * Math.PI * 2;
    innerAngles.set(id, angle);
    const node = byId.get(id) as PositionedGraphNode;
    node.layer = 1;
    node.x += Math.cos(angle) * INNER_RING_RADIUS;
    node.y += Math.sin(angle) * INNER_RING_RADIUS;
  });

  const owner = new Map(inner.map((id) => [id, id]));
  const branchQueue = [...inner];
  while (branchQueue.length) {
    const current = branchQueue.shift() as string;
    (outgoing.get(current) || []).forEach((target) => {
      if (target === root.id || owner.has(target)) return;
      owner.set(target, owner.get(current) as string);
      branchQueue.push(target);
    });
  }
  const normalizeAngle = (angle?: number) =>
    angle === undefined
      ? Number.POSITIVE_INFINITY
      : (angle - startAngle + Math.PI * 2) % (Math.PI * 2);
  const outer = nodes.filter(
    (node) => node.id !== root.id && !innerSet.has(node.id),
  );
  outer.sort(
    (left, right) =>
      normalizeAngle(innerAngles.get(owner.get(left.id) || "")) -
        normalizeAngle(innerAngles.get(owner.get(right.id) || "")) ||
      right.degree - left.degree ||
      left.id.localeCompare(right.id),
  );
  outer.forEach((node, index) => {
    const angle =
      startAngle + (index / Math.max(1, outer.length)) * Math.PI * 2;
    node.x += Math.cos(angle) * OUTER_RING_RADIUS;
    node.y += Math.sin(angle) * OUTER_RING_RADIUS;
  });
  return { nodes, byId };
}

export function reciprocalEdgeKeys(edges: GraphSnapshotEdge[]): Set<string> {
  const keys = new Set(
    edges.map((edge) => `${edge.source}\u0000${edge.target}`),
  );
  return new Set(
    [...keys].filter((key) => {
      const [source, target] = key.split("\u0000");
      return keys.has(`${target}\u0000${source}`);
    }),
  );
}

export function edgePath(
  edge: GraphSnapshotEdge,
  byId: Map<string, PositionedGraphNode>,
  reciprocal: Set<string>,
): string {
  const source = byId.get(edge.source);
  const target = byId.get(edge.target);
  if (!source || !target) return "";
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const startX = source.x + (dx / distance) * (source.radius + 2);
  const startY = source.y + (dy / distance) * (source.radius + 2);
  const endX = target.x - (dx / distance) * (target.radius + 6);
  const endY = target.y - (dy / distance) * (target.radius + 6);
  if (!reciprocal.has(`${edge.source}\u0000${edge.target}`))
    return `M ${startX} ${startY} L ${endX} ${endY}`;
  const curve = edge.source.localeCompare(edge.target) < 0 ? 16 : -16;
  const midX = (startX + endX) / 2 - (dy / distance) * curve;
  const midY = (startY + endY) / 2 + (dx / distance) * curve;
  return `M ${startX} ${startY} Q ${midX} ${midY} ${endX} ${endY}`;
}
