import { ReMeClient } from "../core/client.js";
import { formatReMeContext } from "../core/context.js";
import { OPENCLAW_CONFIG_SCHEMA, resolveOpenClawConfig } from "./config.js";
import type { OpenClawPluginDefinition } from "./host.js";
import { OpenClawReMeRuntime } from "./runtime.js";
import { registerOpenClawTools } from "./tools.js";

const plugin: OpenClawPluginDefinition = {
  id: "reme",
  name: "ReMe",
  description: "ReMe file-native long-term memory",
  kind: "memory",
  configSchema: {
    jsonSchema: OPENCLAW_CONFIG_SCHEMA,
    parse: (value) => resolveOpenClawConfig(asConfig(value)),
  },
  register(api) {
    const config = resolveOpenClawConfig(api.pluginConfig);
    const client = new ReMeClient(config);
    const runtime = new OpenClawReMeRuntime(client, config, api.logger);
    registerOpenClawTools(api, client, config);

    if (config.autoRecall) {
      api.on("before_agent_start", async (event, context) => {
        if (!capturesTrigger(context.trigger)) return;
        const query = event.prompt.trim();
        if (!query) return;
        const result = await client.search(query, {
          limit: config.recallLimit,
          minScore: config.recallMinScore,
        });
        if (!result.ok) {
          api.logger.warn(
            `[reme] openclaw_recall_failed: ${result.error || "unknown error"}`,
          );
          return;
        }
        const prependContext = formatReMeContext(result.answer);
        return prependContext ? { prependContext } : undefined;
      });
    }

    api.on("agent_end", (event, context) => {
      if (event.success) runtime.capture(event.messages, context);
    });

    api.registerService({
      id: "reme",
      start: () => api.logger.info(`[reme] connected to ${config.endpoint}`),
      stop: () => runtime.dispose(),
    });
  },
};

export default plugin;
export { OPENCLAW_CONFIG_SCHEMA, resolveOpenClawConfig } from "./config.js";
export { captureLastTurn, openClawSessionId } from "./messages.js";
export { OpenClawReMeRuntime } from "./runtime.js";

function asConfig(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

function capturesTrigger(trigger: string | undefined): boolean {
  return trigger === undefined || trigger === "user";
}
