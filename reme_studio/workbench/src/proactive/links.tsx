import {
  useEffect,
  useRef,
  useState,
  useId,
  type RefObject,
  type CSSProperties,
} from "react";
import { curve, isTimeAction, sourceRoute } from "./flow.ts";
import type {
  ProactiveOverview,
  OverviewTransition,
} from "../../../client/src/wire.ts";

export type Flight = {
  action: OverviewTransition;
  edge: string;
  expires: number;
  delay: number;
};
export default function FlowLinks({
  stage,
  nodes,
  overview,
  flights,
  labels,
  activeSources,
}: {
  stage: RefObject<HTMLDivElement | null>;
  nodes: RefObject<Map<string, HTMLElement>>;
  overview: ProactiveOverview;
  flights: Flight[];
  labels: { read: string; write: string };
  activeSources?: string[];
}) {
  const [paths, setPaths] = useState<Record<string, string>>({});
  const [ports, setPorts] = useState<{ x: number; y: number } | null>(null);
  const id = useId().replace(/[^a-z0-9_-]/gi, "");
  const current = useRef(overview);
  current.current = overview;
  // Parent and sibling DOM refs are all attached before passive effects run.
  useEffect(() => {
    const root = stage.current;
    if (!root) return;
    let frame = 0;
    const measure = () => {
      const bounds = root.getBoundingClientRect();
      const mobile = bounds.width < 760;
      const rect = (key: string) => {
        const value = nodes.current.get(key)?.getBoundingClientRect();
        return value
          ? {
              left: value.left - bounds.left,
              right: value.right - bounds.left,
              top: value.top - bounds.top,
              bottom: value.bottom - bounds.top,
              x: value.left - bounds.left + value.width / 2,
              y: value.top - bounds.top + value.height / 2,
            }
          : null;
      };
      const result: Record<string, string> = {};
      const connect = (key: string, from: string, to: string) => {
        const a = rect(from),
          b = rect(to);
        if (!a || !b) return;
        if (mobile && key === "deliver") {
          const gutter = Math.max(a.right, b.right) + 13;
          result[key] =
            `M${a.right},${a.y} C${gutter},${a.y} ${gutter},${a.y} ${gutter},${a.y + 24} L${gutter},${b.y - 24} Q${gutter},${b.y} ${b.right},${b.y}`;
        } else
          result[key] = curve(
            mobile ? { x: a.x, y: a.bottom } : { x: a.right, y: a.y },
            mobile ? { x: b.x, y: b.top } : { x: b.left, y: b.y },
            mobile,
          );
      };
      const inbox = rect("inbox"),
        sourceList = rect("source-list");
      for (const source of current.current.sources) {
        const key = `source:${source.source_id}`;
        const origin = rect(key);
        const timed = source.trigger_type === "time";
        if (mobile) {
          const group = rect("sources");
          if (group && inbox)
            result[key] = curve(
              { x: group.x + (timed ? 7 : -7), y: group.bottom },
              { x: inbox.x + (timed ? 7 : -7), y: inbox.top },
              true,
            );
        } else if (
          origin &&
          inbox &&
          (!sourceList ||
            (origin.y >= sourceList.top && origin.y <= sourceList.bottom))
        ) {
          result[key] = sourceRoute(
            { x: origin.right, y: origin.y },
            { x: inbox.left, y: inbox.y + (timed ? 6 : -6) },
            inbox.left - (timed ? 16 : 30),
          );
        }
      }
      connect("claim", "inbox", "runtime");
      connect("deliver", "runtime", "remy");
      connect("remind", "remy", "reminders");
      const agent = rect("runtime"),
        workspace = rect("workspace");
      if (agent && workspace) {
        const port = { x: agent.x, y: (agent.bottom + workspace.top) / 2 };
        setPorts((previous) =>
          previous?.x === port.x && previous?.y === port.y ? previous : port,
        );
        for (const [key, offset] of [
          ["write", 19],
          ["read", -19],
        ] as const) {
          const top = { x: agent.x + offset, y: agent.bottom },
            bottom = { x: agent.x + offset, y: workspace.top };
          result[key] =
            key === "write"
              ? curve(top, bottom, true)
              : `M${bottom.x},${bottom.y} C${bottom.x},${bottom.y - 25} ${top.x},${top.y + 25} ${top.x},${top.y}`;
        }
      }
      setPaths((previous) =>
        JSON.stringify(previous) === JSON.stringify(result) ? previous : result,
      );
    };
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(root);
    for (const element of nodes.current.values()) observer.observe(element);
    window.addEventListener("resize", schedule);
    root.addEventListener("scroll", schedule, true);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("resize", schedule);
      root.removeEventListener("scroll", schedule, true);
    };
  }, [
    stage,
    nodes,
    overview.sources.map((source) => source.source_id).join("|"),
  ]);
  return (
    <svg className="flow-links" aria-hidden="true">
      <defs>
        <marker
          id={`${id}-port`}
          markerWidth="8"
          markerHeight="8"
          refX="4"
          refY="4"
          markerUnits="userSpaceOnUse"
        >
          <circle
            cx="4"
            cy="4"
            r="2.7"
            fill="var(--surface-opaque)"
            stroke="currentColor"
            strokeWidth="1.4"
          />
        </marker>
        <marker
          id={`${id}-arrow`}
          markerWidth="7"
          markerHeight="7"
          refX="6"
          refY="3"
          orient="auto"
        >
          <path
            d="M1,1 L6,3 L1,5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
          />
        </marker>
        <filter
          id={`${id}-glow`}
          x="-100%"
          y="-100%"
          width="300%"
          height="300%"
        >
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      {Object.entries(paths).map(([key, path]) => (
        <path
          key={key}
          className="flow-track"
          data-dim={
            key.startsWith("source:") &&
            activeSources !== undefined &&
            !activeSources.includes(key.slice(7))
          }
          data-time={
            key.startsWith("source:") &&
            overview.sources.some(
              (source) =>
                `source:${source.source_id}` === key &&
                source.trigger_type === "time",
            )
          }
          d={path}
          markerStart={`url(#${id}-port)`}
          markerEnd={`url(#${id}-arrow)`}
        />
      ))}
      {flights.map(
        ({ action, edge, delay }) =>
          paths[edge] && (
            <g
              className="flow-flight"
              key={action.id}
              style={{ "--flight-delay": `${delay}ms` } as CSSProperties}
              data-time={isTimeAction(action, overview)}
            >
              <path
                d={paths[edge]}
                className="flow-comet-glow"
                filter={`url(#${id}-glow)`}
                pathLength="1"
              />
              <path d={paths[edge]} className="flow-comet" pathLength="1" />
              <circle
                r="3"
                fill="currentColor"
                className="flow-comet-head"
                style={{ offsetPath: `path("${paths[edge]}")` }}
              />
            </g>
          ),
      )}
      {ports && (
        <g className="flow-link-labels">
          <text x={ports.x - 32} y={ports.y + 3} textAnchor="end">
            {labels.read}
          </text>
          <text x={ports.x + 32} y={ports.y + 3}>
            {labels.write}
          </text>
        </g>
      )}
    </svg>
  );
}
