import { useEffect, useId, useRef, useState } from "react";
import type { ProactiveOverview } from "../../../client/src/wire.ts";
import {
  isTimeAction,
  relayTiming,
  transferOrigin,
  type Flight,
} from "./flow.ts";

/** A quiet clock-like drift is decorative; bright comets represent confirmed handoffs. */
export default function FlowTransfers({
  flights,
  overview,
  live,
}: {
  flights: Flight[];
  overview: ProactiveOverview;
  live: boolean;
}) {
  const root = useRef<HTMLDivElement>(null);
  const gradient = useId();
  const [anchors, setAnchors] = useState<{ top: number; left: number }[]>([]);
  useEffect(() => {
    const stage = root.current!.parentElement!;
    const targets = [
      ...stage.querySelectorAll<HTMLElement>(".flow-node-header"),
    ];
    const measure = () => {
      const origin = stage.getBoundingClientRect();
      const scaleX = origin.width / stage.offsetWidth || 1;
      const scaleY = origin.height / stage.offsetHeight || 1;
      const next = targets.map((target) => {
        const rect = target.getBoundingClientRect();
        return {
          top: (rect.top - origin.top) / scaleY,
          left: (rect.left - origin.left) / scaleX,
        };
      });
      setAnchors((previous) =>
        next.every(
          (point, index) =>
            point.top === previous[index]?.top &&
            point.left === previous[index]?.left,
        )
          ? previous
          : next,
      );
    };
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    targets.forEach((target) => observer.observe(target));
    measure();
    return () => observer.disconnect();
  }, []);
  const point = (index: number) => ({
    x: anchors[index].left + 22,
    y: anchors[index].top - 1,
  });
  const track = (from: number, to: number) => {
    const start = point(from);
    const end = point(to);
    return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
  };
  return (
    <div className="flow-transfers" ref={root} data-live={live} aria-hidden>
      {anchors.length === 5 && (
        <>
          <svg className="flow-timeline">
            <defs>
              <linearGradient
                id={gradient}
                gradientUnits="userSpaceOnUse"
                x1={point(0).x}
                y1={point(0).y}
                x2={point(4).x}
                y2={point(4).y}
              >
                <stop offset="0%" stopColor="var(--flow-blue)" />
                <stop offset="68%" stopColor="var(--flow-purple)" />
                <stop offset="100%" stopColor="var(--flow-green)" />
              </linearGradient>
            </defs>
            <path
              className="flow-timeline-track"
              d={track(0, 4)}
              stroke={`url(#${gradient})`}
            />
            {anchors.map((_, index) => {
              const { x, y } = point(index);
              const working =
                live &&
                overview.agents.some(
                  (agent) =>
                    (index === 2
                      ? agent.agent_type !== "chat"
                      : index === 3 && agent.agent_type === "chat") &&
                    (agent.active_run || agent.status === "active"),
                );
              return (
                <g
                  className="flow-timeline-node"
                  key={index}
                  data-step={index}
                  data-working={working}
                >
                  <circle
                    className="flow-timeline-orbit"
                    cx={x}
                    cy={y}
                    r={10}
                  />
                  <circle className="flow-timeline-seat" cx={x} cy={y} r={5} />
                  <circle className="flow-timeline-dot" cx={x} cy={y} r={3} />
                </g>
              );
            })}
            {flights.map((flight) => {
              const from = transferOrigin(flight.edge);
              const start = point(from);
              const end = point(from + 1);
              return (
                <g
                  className="flow-timeline-pings"
                  key={flight.action.id}
                  data-origin={from}
                  data-time={isTimeAction(flight.action, overview)}
                >
                  <circle
                    className="flow-timeline-ping"
                    cx={start.x}
                    cy={start.y}
                    r={5}
                    style={{ animationDelay: `${flight.delay}ms` }}
                  />
                  <circle
                    className="flow-timeline-ping"
                    cx={end.x}
                    cy={end.y}
                    r={5}
                    style={{
                      animationDelay: `${flight.delay + relayTiming.arrive}ms`,
                    }}
                  />
                  <circle
                    className="flow-timeline-touch"
                    cx={end.x}
                    cy={end.y}
                    r={3}
                    style={{
                      animationDelay: `${flight.delay + relayTiming.arrive}ms`,
                    }}
                  />
                </g>
              );
            })}
          </svg>
          {[0, 1].map((index) => (
            <span
              className="flow-timeline-ambient"
              key={index}
              data-tone={index}
              style={{
                offsetPath: `path("${track(0, 4)}")`,
                animationDelay: `${index * -4}s`,
              }}
            />
          ))}
          {flights.map((flight) => {
            const from = transferOrigin(flight.edge);
            return (
              <span
                className="flow-comet"
                key={flight.action.id}
                data-origin={from}
                data-time={isTimeAction(flight.action, overview)}
                style={{
                  offsetPath: `path("${track(from, from + 1)}")`,
                  animationDelay: `${flight.delay + relayTiming.depart}ms`,
                  animationDuration: `${relayTiming.arrive - relayTiming.depart}ms`,
                }}
              />
            );
          })}
        </>
      )}
    </div>
  );
}
