"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  CircleAlert,
  ExternalLink,
  Link2,
  LoaderCircle,
  Maximize2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { getAppConfig, getGraphSnapshot, readWorkspaceFile } from "../api";
import { useI18n } from "../i18n";
import { useWorkspaceStore } from "../store";
import type { GraphSnapshot, MemoryGraphRoot } from "../types";
import {
  edgePath,
  GRAPH_HEIGHT,
  GRAPH_WIDTH,
  graphBelowRoot,
  INNER_RING_RADIUS,
  layoutGraph,
  nodeLabel,
  OUTER_RING_RADIUS,
  reciprocalEdgeKeys,
  shortNodeLabel,
  type PositionedGraphNode,
} from "./memory-graph";
import styles from "./memory-graph.module.css";

const AUTO_REFRESH_MS = 10_000;

interface NodeOffset {
  x: number;
  y: number;
}
interface DragSession {
  pointerId: number;
  nodeId: string;
  start: NodeOffset;
  base: NodeOffset;
}

function pointerPosition(
  event: ReactPointerEvent<SVGGElement>,
  zoom: number,
): NodeOffset {
  const bounds = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
  if (!bounds?.width || !bounds.height)
    return { x: event.clientX, y: event.clientY };
  const viewX = ((event.clientX - bounds.left) / bounds.width) * GRAPH_WIDTH;
  const viewY = ((event.clientY - bounds.top) / bounds.height) * GRAPH_HEIGHT;
  return {
    x: GRAPH_WIDTH / 2 + (viewX - GRAPH_WIDTH / 2) / zoom,
    y: GRAPH_HEIGHT / 2 + (viewY - GRAPH_HEIGHT / 2) / zoom,
  };
}

export default function MemoryGraphView({ root }: { root: MemoryGraphRoot }) {
  const { t } = useI18n();
  const [snapshot, setSnapshot] = useState<GraphSnapshot>();
  const [dailyDirectory, setDailyDirectory] = useState("daily");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedId, setSelectedId] = useState("");
  const [hoveredId, setHoveredId] = useState("");
  const [zoom, setZoom] = useState(1);
  const [offsets, setOffsets] = useState<Record<string, NodeOffset>>({});
  const [draggingId, setDraggingId] = useState("");
  const dragSession = useRef<DragSession | undefined>(undefined);
  const didDrag = useRef(false);
  const tabs = useWorkspaceStore((state) => state.tabs);
  const openMarkdown = useWorkspaceStore((state) => state.openMarkdown);
  const hydrateMarkdown = useWorkspaceStore((state) => state.hydrateMarkdown);
  const failMarkdown = useWorkspaceStore((state) => state.failMarkdown);

  useEffect(() => {
    let mounted = true;
    let fetching = false;
    const load = async () => {
      if (!mounted || fetching) return;
      fetching = true;
      try {
        const [next, config] = await Promise.all([
          getGraphSnapshot(),
          getAppConfig(),
        ]);
        if (!mounted) return;
        setSnapshot(next);
        setDailyDirectory(config.daily_dir);
        setError(false);
        setSelectedId((current) =>
          next.nodes.some((node) => node.id === current) ? current : "",
        );
      } catch {
        if (mounted) setError(true);
      } finally {
        fetching = false;
        if (mounted) setLoading(false);
      }
    };
    void load();
    const timer = window.setInterval(() => {
      if (!document.hidden) void load();
    }, AUTO_REFRESH_MS);
    const onVisibilityChange = () => {
      if (!document.hidden) void load();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      mounted = false;
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  const graphSnapshot = useMemo(
    () =>
      snapshot ? graphBelowRoot(snapshot, root, dailyDirectory) : undefined,
    [dailyDirectory, root, snapshot],
  );
  const baseGraph = useMemo(
    () => layoutGraph(graphSnapshot || { version: 1, nodes: [], edges: [] }),
    [graphSnapshot],
  );
  const graph = useMemo(() => {
    const nodes = baseGraph.nodes.map((node) => ({
      ...node,
      x: node.x + (offsets[node.id]?.x || 0),
      y: node.y + (offsets[node.id]?.y || 0),
    }));
    return { nodes, byId: new Map(nodes.map((node) => [node.id, node])) };
  }, [baseGraph, offsets]);
  const selected = graphSnapshot?.nodes.find((node) => node.id === selectedId);
  const activeId = hoveredId || selectedId;
  const inbound =
    graphSnapshot?.edges.filter((edge) => edge.target === selectedId) || [];
  const outbound =
    graphSnapshot?.edges.filter((edge) => edge.source === selectedId) || [];
  const neighbors = useMemo(() => {
    const result = new Set<string>();
    graphSnapshot?.edges.forEach((edge) => {
      if (edge.source === activeId) result.add(edge.target);
      if (edge.target === activeId) result.add(edge.source);
    });
    return result;
  }, [activeId, graphSnapshot]);
  const reciprocal = useMemo(
    () => reciprocalEdgeKeys(graphSnapshot?.edges || []),
    [graphSnapshot],
  );
  const labelIds = useMemo(() => {
    const limit = Math.min(
      10,
      Math.max(5, Math.ceil(Math.sqrt(graph.nodes.length) * 1.5)),
    );
    return new Set(
      [...graph.nodes]
        .sort(
          (left, right) =>
            Number(right.virtual) - Number(left.virtual) ||
            right.degree - left.degree ||
            left.id.localeCompare(right.id),
        )
        .slice(0, limit)
        .map((node) => node.id),
    );
  }, [graph.nodes]);

  const openFile = async (node: PositionedGraphNode) => {
    if (!node.indexed || node.virtual) return;
    const existing = tabs.some(
      (tab) => tab.type === "markdown" && tab.path === node.path,
    );
    const id = openMarkdown(node.path);
    if (existing) return;
    try {
      const file = await readWorkspaceFile(node.path);
      hydrateMarkdown(id, file.content, file.stat.mtime);
    } catch (openError) {
      failMarkdown(
        id,
        openError instanceof Error ? openError.message : t("fileReadFailed"),
      );
    }
  };

  const startDrag = (event: ReactPointerEvent<SVGGElement>, nodeId: string) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    dragSession.current = {
      pointerId: event.pointerId,
      nodeId,
      start: pointerPosition(event, zoom),
      base: offsets[nodeId] || { x: 0, y: 0 },
    };
    didDrag.current = false;
    setDraggingId(nodeId);
    event.currentTarget.setPointerCapture(event.pointerId);
  };
  const moveDrag = (event: ReactPointerEvent<SVGGElement>) => {
    const drag = dragSession.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const pointer = pointerPosition(event, zoom);
    const dx = pointer.x - drag.start.x;
    const dy = pointer.y - drag.start.y;
    if (Math.hypot(dx, dy) > 2) didDrag.current = true;
    setOffsets((current) => ({
      ...current,
      [drag.nodeId]: { x: drag.base.x + dx, y: drag.base.y + dy },
    }));
  };
  const endDrag = (event: ReactPointerEvent<SVGGElement>) => {
    if (dragSession.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
    dragSession.current = undefined;
    setDraggingId("");
  };

  if (loading && !snapshot)
    return (
      <div className={styles.state}>
        <LoaderCircle className="spin" size={18} />
        {t("memoryGraphLoading")}
      </div>
    );
  if (error && !snapshot)
    return (
      <div className={`${styles.state} ${styles.error}`}>
        <CircleAlert size={18} />
        {t("memoryGraphLoadFailed")}
      </div>
    );

  return (
    <section className={styles.view} aria-label={t("memoryGraph")}>
      <header className={styles.toolbar}>
        <div>
          <strong>
            {t("memoryGraph")} · {root}
          </strong>
          <span>
            {t("memoryGraphCounts", {
              nodes: String(graphSnapshot?.nodes.length || 0),
              edges: String(graphSnapshot?.edges.length || 0),
            })}
          </span>
        </div>
        <div className={styles.actions}>
          <button
            onClick={() => setZoom((current) => Math.max(0.7, current - 0.15))}
            aria-label={t("memoryGraphZoomOut")}
          >
            <ZoomOut size={15} />
          </button>
          <button
            onClick={() => setZoom((current) => Math.min(1.6, current + 0.15))}
            aria-label={t("memoryGraphZoomIn")}
          >
            <ZoomIn size={15} />
          </button>
          <button
            onClick={() => {
              setZoom(1);
              setOffsets({});
            }}
            aria-label={t("memoryGraphFit")}
          >
            <Maximize2 size={15} />
          </button>
        </div>
      </header>
      {!graph.nodes.length ? (
        <div className={styles.state}>
          <Link2 size={18} />
          {t("memoryGraphEmpty")}
        </div>
      ) : (
        <div
          className={`${styles.content} ${selected ? styles.withDetails : ""}`}
        >
          <div className={styles.canvas}>
            <svg viewBox={`0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`} role="img">
              <title>{t("memoryGraph")}</title>
              <defs>
                <marker
                  id={`graph-arrow-${root}`}
                  viewBox="0 0 10 10"
                  refX="9"
                  refY="5"
                  markerWidth="5"
                  markerHeight="5"
                  orient="auto"
                >
                  <path d="M 0 0 L 10 5 L 0 10 z" />
                </marker>
              </defs>
              <g
                transform={`translate(${GRAPH_WIDTH / 2} ${
                  GRAPH_HEIGHT / 2
                }) scale(${zoom}) translate(${-GRAPH_WIDTH / 2} ${
                  -GRAPH_HEIGHT / 2
                })`}
              >
                <g className={styles.orbits} aria-hidden="true">
                  <circle
                    cx={GRAPH_WIDTH / 2}
                    cy={GRAPH_HEIGHT / 2}
                    r={INNER_RING_RADIUS}
                  />
                  {graph.nodes.some((node) => node.layer === 2) && (
                    <circle
                      cx={GRAPH_WIDTH / 2}
                      cy={GRAPH_HEIGHT / 2}
                      r={OUTER_RING_RADIUS}
                    />
                  )}
                </g>
                <g className={styles.edges}>
                  {graphSnapshot?.edges.map((edge) => {
                    const related =
                      edge.source === activeId || edge.target === activeId;
                    return (
                      <path
                        key={`${edge.source}:${edge.target}:${
                          edge.target_anchor || ""
                        }`}
                        d={edgePath(edge, graph.byId, reciprocal)}
                        className={`${related ? styles.edgeActive : ""} ${
                          activeId && !related ? styles.muted : ""
                        }`}
                        markerEnd={`url(#graph-arrow-${root})`}
                      >
                        <title>
                          {edge.source} → {edge.target}
                          {edge.target_anchor ? `#${edge.target_anchor}` : ""}
                        </title>
                      </path>
                    );
                  })}
                </g>
                <g>
                  {graph.nodes.map((node) => {
                    const active = node.id === activeId;
                    const related = neighbors.has(node.id);
                    return (
                      <g
                        key={node.id}
                        className={`${styles.node} ${
                          node.virtual ? styles.root : ""
                        } ${node.degree >= 5 ? styles.hub : ""} ${
                          active ? styles.active : ""
                        } ${related ? styles.related : ""} ${
                          activeId && !active && !related ? styles.muted : ""
                        } ${draggingId === node.id ? styles.dragging : ""}`}
                        style={{
                          transform: `translate(${node.x}px, ${node.y}px)`,
                        }}
                        role="button"
                        tabIndex={0}
                        aria-label={nodeLabel(node)}
                        onClick={(event) => {
                          event.stopPropagation();
                          if (didDrag.current) {
                            didDrag.current = false;
                            return;
                          }
                          setSelectedId(node.id);
                        }}
                        onDoubleClick={() => void openFile(node)}
                        onPointerDown={(event) => startDrag(event, node.id)}
                        onPointerMove={moveDrag}
                        onPointerUp={endDrag}
                        onPointerCancel={endDrag}
                        onMouseEnter={() => setHoveredId(node.id)}
                        onMouseLeave={() => setHoveredId("")}
                        onFocus={() => setHoveredId(node.id)}
                        onBlur={() => setHoveredId("")}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            setSelectedId(node.id);
                          }
                        }}
                      >
                        <circle className={styles.halo} r={node.radius + 6} />
                        <circle className={styles.dot} r={node.radius} />
                        <title>{node.path}</title>
                      </g>
                    );
                  })}
                </g>
                <g className={styles.labels}>
                  {graph.nodes
                    .filter(
                      (node) => labelIds.has(node.id) || node.id === activeId,
                    )
                    .map((node) => (
                      <text
                        key={node.id}
                        className={
                          node.id === activeId ? styles.activeLabel : ""
                        }
                        x={node.x}
                        y={node.y + node.radius + 20}
                        textAnchor="middle"
                      >
                        {shortNodeLabel(node)}
                      </text>
                    ))}
                </g>
              </g>
            </svg>
            <div className={styles.legend}>
              <span>
                <i />
                {t("memoryGraphIndexed")}
              </span>
              <span>
                <b>→</b>
                {t("memoryGraphDirection")}
              </span>
            </div>
          </div>
          {selected && (
            <aside className={styles.details}>
              <span className={styles.status}>
                {selected.virtual ? root : t("memoryGraphIndexed")}
              </span>
              <h2>{nodeLabel(selected)}</h2>
              <code>{selected.path}</code>
              {selected.description && <p>{selected.description}</p>}
              {selected.indexed && !selected.virtual && (
                <button
                  className={styles.openFile}
                  onClick={() =>
                    void openFile(
                      graph.byId.get(selected.id) as PositionedGraphNode,
                    )
                  }
                >
                  <span>{t("memoryGraphOpenFile")}</span>
                  <ExternalLink size={14} />
                </button>
              )}
              <div className={styles.links}>
                <strong>
                  {t("memoryGraphOutbound", { count: String(outbound.length) })}
                </strong>
                {outbound.map((edge) => (
                  <button
                    key={`${edge.target}:${edge.target_anchor || ""}`}
                    onClick={() => setSelectedId(edge.target)}
                  >
                    <span>
                      {nodeLabel(
                        graph.byId.get(edge.target) || {
                          ...selected,
                          name: "",
                          path: edge.target,
                        },
                      )}
                    </span>
                    <small>
                      {edge.target_anchor ? `#${edge.target_anchor}` : "→"}
                    </small>
                  </button>
                ))}
              </div>
              <div className={styles.links}>
                <strong>
                  {t("memoryGraphInbound", { count: String(inbound.length) })}
                </strong>
                {inbound.map((edge) => (
                  <button
                    key={`${edge.source}:${edge.target_anchor || ""}`}
                    onClick={() => setSelectedId(edge.source)}
                  >
                    <span>
                      {nodeLabel(
                        graph.byId.get(edge.source) || {
                          ...selected,
                          name: "",
                          path: edge.source,
                        },
                      )}
                    </span>
                    <small>←</small>
                  </button>
                ))}
              </div>
            </aside>
          )}
        </div>
      )}
    </section>
  );
}
