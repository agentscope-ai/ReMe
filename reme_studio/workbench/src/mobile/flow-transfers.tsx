import { useEffect, useRef, useState } from "react";
import type { ProactiveOverview } from "../../../client/src/wire.ts";
import {
  isTimeAction,
  relayTiming,
  transferOrigin,
  type Flight,
} from "../proactive/flow.ts";
import { verticalConnection, type FlowBox } from "./flow-connections.ts";

export default function MobileFlowTransfers({
  flights,
  overview,
}: {
  flights: Flight[];
  overview: ProactiveOverview;
}) {
  const root = useRef<SVGSVGElement>(null);
  const [boxes, setBoxes] = useState<FlowBox[]>([]);
  useEffect(() => {
    const stage = root.current!.parentElement!;
    const cards = [...stage.querySelectorAll<HTMLElement>(".flow-node")];
    const measure = () => {
      const origin = stage.getBoundingClientRect();
      const scaleX = origin.width / stage.offsetWidth || 1;
      const scaleY = origin.height / stage.offsetHeight || 1;
      const next = cards.map((card) => {
        const rect = card.getBoundingClientRect();
        return {
          top: (rect.top - origin.top) / scaleY,
          bottom: (rect.bottom - origin.top) / scaleY,
          left: (rect.left - origin.left) / scaleX,
          right: (rect.right - origin.left) / scaleX,
        };
      });
      setBoxes((previous) =>
        JSON.stringify(previous) === JSON.stringify(next) ? previous : next,
      );
    };
    const observer = new ResizeObserver(measure);
    observer.observe(stage);
    cards.forEach((card) => observer.observe(card));
    measure();
    return () => observer.disconnect();
  }, []);
  const connection = (index: number) =>
    verticalConnection(boxes[index], boxes[index + 1]);
  return (
    <svg className="mobile-flow-transfers" ref={root} aria-hidden>
      {boxes.length === 5 && (
        <>
          {boxes.slice(0, -1).map((_, index) => (
            <g className="mobile-flow-bridge" key={index}>
              {connection(index).chevrons.map((path, chevron) => (
                <path key={chevron} d={path} />
              ))}
            </g>
          ))}
          {flights.map((flight) => {
            const from = transferOrigin(flight.edge);
            return (
              <path
                className="mobile-flow-transfer"
                key={flight.action.id}
                data-origin={from}
                data-time={isTimeAction(flight.action, overview)}
                d={connection(from).path}
                pathLength={100}
                style={{
                  animationDelay: `${flight.delay + relayTiming.depart}ms`,
                  animationDuration: `${relayTiming.arrive - relayTiming.depart}ms`,
                }}
              />
            );
          })}
        </>
      )}
    </svg>
  );
}
