import type { CSSProperties } from "react";
import { relayTiming, transferOrigin, type Flight } from "./flow.ts";

/** One transfer lights its sender first, then its receiver after the gap crossing. */
export default function FlowRelay({
  flights,
  step,
  kind = "row",
}: {
  flights: Flight[];
  step: number;
  kind?: "row" | "header";
}) {
  return flights.map((flight) => {
    const from = transferOrigin(flight.edge);
    if (step !== from && step !== from + 1) return null;
    const incoming = step === from + 1;
    return (
      <span
        key={flight.action.id}
        className="flow-relay"
        data-kind={kind}
        data-incoming={incoming}
        aria-hidden
        style={
          {
            "--relay-delay": `${flight.delay + (incoming ? relayTiming.arrive : 0)}ms`,
            "--relay-duration": `${incoming ? relayTiming.receive : relayTiming.depart}ms`,
            "--relay-edge-delay": `${flight.delay + (incoming ? relayTiming.arrive : relayTiming.depart - 140)}ms`,
          } as CSSProperties
        }
      />
    );
  });
}
