import {
  buildJsonPluginConfigSchema,
  definePluginEntry,
} from "openclaw/plugin-sdk/plugin-entry";
import type { OpenClawPluginDefinition } from "openclaw/plugin-sdk/plugin-entry";

import { ReMeClient } from "../core/client.js";
import { formatReMeContext } from "../core/context.js";
import { memoryGuidance } from "../core/guidance.js";
import {
  OPENCLAW_CONFIG_SCHEMA,
  OPENCLAW_CONFIG_UI_HINTS,
  resolveOpenClawConfig,
} from "./config.js";
import { OpenClawReMeRuntime } from "./runtime.js";
import { registerOpenClawTools } from "./tools.js";

/** Current OpenClaw entrypoint: manifest-owned kind plus SDK-owned contracts. */
const plugin: OpenClawPluginDefinition = definePluginEntry({
  id: "reme",
  name: "ReMe",
  description: "ReMe file-native long-term memory",
  configSchema: buildJsonPluginConfigSchema(OPENCLAW_CONFIG_SCHEMA, {
    uiHints: OPENCLAW_CONFIG_UI_HINTS,
  }),
  register(api) {
    const config = resolveOpenClawConfig(api.pluginConfig);
    const client = new ReMeClient(config);
    const runtime = new OpenClawReMeRuntime(client, config, api.logger);
    registerOpenClawTools(api, client, config);

    // before_prompt_build is the current prompt-mutation hook. Keeping recall
    // here prevents ReMe context from leaking into the captured user message.
    api.on(
      "before_prompt_build",
      async (event, context) => {
        if (!runtime.accepts(context)) return;
        runtime.rememberPrompt(event.prompt, context);
        const guidance = memoryGuidance(config.language);
        if (!config.autoRecall) return { prependSystemContext: guidance };
        const query = event.prompt.trim();
        if (!query) return { prependSystemContext: guidance };
        const result = await client.search(query, {
          limit: config.searchLimit,
          minScore: config.recallMinScore,
        });
        if (!result.ok) {
          api.logger.warn(
            `[reme] openclaw_recall_failed: ${result.error || "unknown error"}`,
          );
          return { prependSystemContext: guidance };
        }
        const prependContext = formatReMeContext(result.answer);
        return {
          prependSystemContext: guidance,
          ...(prependContext ? { prependContext } : {}),
        };
      },
      { timeoutMs: config.requestTimeoutMs },
    );

    api.on("agent_end", (event, context) => {
      const prompt = runtime.takePrompt(context);
      if (event.success) runtime.capture(event.messages, context, prompt);
    });
    api.on("session_end", (_event, context) => runtime.disposeSession(context));

    api.registerService({
      id: "reme",
      start: () => {
        runtime.start();
        api.logger.info(`[reme] connected to ${config.endpoint}`);
      },
      stop: () => runtime.disposeAll(),
    });
  },
});

export default plugin;
export {
  OPENCLAW_CONFIG_SCHEMA,
  OPENCLAW_CONFIG_UI_HINTS,
  resolveOpenClawConfig,
} from "./config.js";
export { captureLastTurn, openClawSessionId } from "./messages.js";
export { OpenClawReMeRuntime } from "./runtime.js";
