import { delay } from "./reme-client.ts";
import type { ReMeClient } from "./reme-client.ts";
import type {
  OverviewEvent,
  OverviewTransition,
  ProactiveOverview,
} from "./wire.ts";

export class OverviewVersionError extends Error {
  constructor() {
    super("Proactive overview requires a new snapshot");
  }
}

function animationsAllowed(): boolean {
  return (
    (typeof document === "undefined" || !document.hidden) &&
    (typeof matchMedia !== "function" ||
      !matchMedia("(prefers-reduced-motion: reduce)").matches)
  );
}

function readOverview(value: unknown): ProactiveOverview {
  const overview = value as ProactiveOverview;
  if (
    !overview ||
    ![
      overview.sources,
      overview.agents,
      overview.reminders,
      overview.workspace,
    ].every(Array.isArray) ||
    !overview.errors ||
    typeof overview.errors !== "object"
  )
    throw new Error("Invalid proactive overview snapshot");
  for (const inbox of [overview.event_inbox, overview.remy_inbox]) {
    if (
      !inbox ||
      !Array.isArray(inbox.items) ||
      !inbox.queues ||
      ![
        inbox.waiting_count,
        inbox.processing_count,
        inbox.processed_count,
      ].every((count) => Number.isSafeInteger(count) && count >= 0)
    )
      throw new Error("Invalid proactive overview inbox");
  }
  return overview;
}

/** One versioned state, with a bounded action identity cache across reconnects. */
export class ProactiveOverviewModel {
  overview: ProactiveOverview | null = null;
  epoch: string | null = null;
  revision = 0;
  asOf: string | null = null;
  live = false;
  private readonly seen = new Set<string>();

  apply(value: unknown): OverviewTransition[] {
    const event = value as OverviewEvent;
    if (event?.type === "heartbeat") return [];
    if (
      !event ||
      !["snapshot", "update"].includes(event.type) ||
      typeof event.epoch !== "string" ||
      !event.epoch ||
      !Number.isSafeInteger(event.revision) ||
      event.revision < 0 ||
      typeof event.as_of !== "string"
    )
      throw new Error("Invalid proactive overview event");
    if (event.type === "snapshot") {
      if (event.epoch === this.epoch && event.revision < this.revision)
        return [];
      const overview = readOverview(event.overview);
      if (this.epoch !== event.epoch) this.seen.clear();
      this.overview = structuredClone(overview);
    } else {
      if (event.epoch !== this.epoch || !this.overview)
        throw new OverviewVersionError();
      if (event.revision <= this.revision) return [];
      if (
        event.previous_revision !== this.revision ||
        !Number.isSafeInteger(event.previous_revision)
      )
        throw new OverviewVersionError();
      if (!Array.isArray(event.changes) || !Array.isArray(event.transitions))
        throw new Error("Invalid proactive overview update");
      const next = structuredClone(this.overview);
      for (const change of event.changes) {
        if (
          ["sources", "agents", "reminders", "workspace"].includes(
            change.collection,
          )
        ) {
          const collection = change.collection as
            "sources" | "agents" | "reminders" | "workspace";
          const key =
            collection === "sources"
              ? "source_id"
              : collection === "agents"
                ? "agent_name"
                : "id";
          const items = next[collection] as unknown as Record<
            string,
            unknown
          >[];
          const index = items.findIndex((item) => item[key] === change.id);
          if (change.value === null) {
            if (index !== -1) items.splice(index, 1);
          } else {
            if (
              typeof change.value !== "object" ||
              !(key in change.value) ||
              change.value[key as keyof typeof change.value] !== change.id
            )
              throw new Error("Invalid proactive overview identity");
            if (index === -1)
              items.push(change.value as unknown as Record<string, unknown>);
            else
              items[index] = change.value as unknown as Record<string, unknown>;
          }
        } else if (
          change.collection === "event_inbox" ||
          change.collection === "remy_inbox"
        ) {
          if (
            !change.value ||
            typeof change.value !== "object" ||
            !("items" in change.value)
          )
            throw new Error("Invalid proactive overview inbox update");
          next[change.collection] = change.value;
        } else if (change.collection === "errors") {
          if (change.value === null) delete next.errors[change.id];
          else if (typeof change.value === "string")
            next.errors[change.id] = change.value;
          else throw new Error("Invalid proactive overview error");
        } else throw new Error("Invalid proactive overview collection");
      }
      readOverview(next);
      next.sources.sort((left, right) =>
        left.source_id.localeCompare(right.source_id),
      );
      next.workspace.sort((left, right) => left.id.localeCompare(right.id));
      next.agents.sort(
        (left, right) =>
          Number(left.agent_type === "chat") -
            Number(right.agent_type === "chat") ||
          left.agent_name.localeCompare(right.agent_name),
      );
      next.reminders.sort(
        (left, right) =>
          Date.parse(right.delivered_at) - Date.parse(left.delivered_at) ||
          left.id.localeCompare(right.id),
      );
      for (const action of event.transitions) {
        if (
          !action ||
          typeof action.id !== "string" ||
          !action.id ||
          typeof action.type !== "string" ||
          typeof action.occurred_at !== "string"
        )
          throw new Error("Invalid proactive overview action");
      }
      this.overview = next;
    }
    this.epoch = event.epoch;
    this.revision = event.revision;
    this.asOf = event.as_of;
    this.live = true;
    if (event.type === "snapshot") return [];
    const actions = event.transitions.filter((action) => {
      if (this.seen.has(action.id)) return false;
      this.seen.add(action.id);
      if (this.seen.size > 2048)
        this.seen.delete(this.seen.values().next().value!);
      return true;
    });
    return actions;
  }
}

/** Keep the last snapshot on disconnect; every new connection calibrates it. */
export async function watchProactiveOverview(
  client: ReMeClient,
  options: {
    agentNames?: string[];
    signal: AbortSignal;
    onChange: (
      model: ProactiveOverviewModel,
      actions: OverviewTransition[],
    ) => void;
    onError?: (error: unknown) => void;
    animations?: () => boolean;
  },
): Promise<void> {
  const model = new ProactiveOverviewModel();
  let retry = 1000;
  while (!options.signal.aborted) {
    try {
      let initialized = false;
      await client.stream(
        "watch_proactive_overview",
        options.agentNames ? { agent_names: options.agentNames } : {},
        options.signal,
        (chunk) => {
          if (chunk.chunk_type === "error")
            throw new Error(String(chunk.chunk));
          if (chunk.chunk_type !== "data") return;
          const event = chunk.chunk as OverviewEvent;
          if (!initialized && event?.type !== "snapshot")
            throw new OverviewVersionError();
          const actions = model.apply(event);
          initialized = true;
          retry = 1000;
          options.onChange(
            model,
            options.animations?.() === false ||
              !animationsAllowed() ||
              actions.length > 24
              ? []
              : actions,
          );
        },
      );
    } catch (error) {
      if (!options.signal.aborted) options.onError?.(error);
    }
    model.live = false;
    options.onChange(model, []);
    if (options.signal.aborted) break;
    try {
      await delay(retry, options.signal);
    } catch {
      break;
    }
    retry = Math.min(30_000, retry * 2);
  }
}
