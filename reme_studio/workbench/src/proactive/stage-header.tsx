import type { ReactNode } from "react";
import { Bell, Boxes, Check, Inbox, Radio, Send } from "lucide-react";
import { translator, type Language } from "../../../client/src/i18n.ts";
import { relayTiming, transferOrigin, type Flight } from "./flow.ts";
import FlowRelay from "./relay.tsx";

const stages = [
  ["flowCollect", "flowSources", Radio],
  ["flowQueue", "flowEventInbox", Inbox],
  ["flowExecute", "flowRuntime", Boxes],
  ["flowHandoff", "flowRemyInbox", Send],
  ["flowNotify", "flowReminders", Bell],
] as const;

/** Each card is a pipeline stage; only confirmed transfers produce flights. */
export default function FlowStageHeader({
  step,
  count,
  subtitle,
  action,
  language,
  flights,
  working = false,
}: {
  step: 0 | 1 | 2 | 3 | 4;
  count: number;
  subtitle: ReactNode;
  action?: ReactNode;
  language: Language;
  flights: Flight[];
  working?: boolean;
}) {
  const t = translator(language);
  const [label, title, Icon] = stages[step];
  const outgoing = flights.some(
    (flight) => transferOrigin(flight.edge) === step,
  );
  return (
    <header
      className="flow-node-header"
      data-step={step + 1}
      data-working={working || undefined}
      data-flowing={outgoing || undefined}
    >
      <FlowRelay flights={flights} step={step} kind="header" />
      <div className="flow-step-label">
        <span className="flow-step-number">
          {String(step + 1).padStart(2, "0")}
        </span>
        {t(label)}
        <i className="flow-stage-status" aria-hidden />
      </div>
      <div className="flow-node-heading">
        <span className="flow-node-icon" aria-hidden>
          <Icon size={20} strokeWidth={1.6} />
          {step === 4 &&
            flights
              .filter((flight) => transferOrigin(flight.edge) === 3)
              .map((flight) => (
                <span
                  className="flow-receipt"
                  key={flight.action.id}
                  style={{
                    animationDelay: `${flight.delay + relayTiming.arrive + 320}ms`,
                  }}
                >
                  <Check size={20} strokeWidth={1.8} />
                </span>
              ))}
        </span>
        <h2>{t(title)}</h2>
        <span className="flow-node-total">{count}</span>
        {action}
      </div>
      <div className="flow-node-subtitle">{subtitle}</div>
    </header>
  );
}
