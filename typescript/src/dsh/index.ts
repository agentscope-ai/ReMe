import { createUserMessage } from "@deepseek-ai/dsh-llm";
import type { Context } from "@deepseek-ai/cordis";
import { installSettingsSection } from "@deepseek-ai/dsh-settings";

import { ReMeClient } from "../core/client.js";
import {
  mergeSettings,
  REME_SETTINGS_NAMESPACE,
  resolveConfig,
  SettingsConfig,
  settingsFrom,
  validateSettings,
} from "./config.js";
import { hasGuidance, memoryGuidance, REME_PLUGIN_SOURCE } from "./guidance.js";
import { ReMeRuntime } from "./runtime.js";
import { ReMeStatusGateway } from "./status-gateway.js";
import { registerReMeTools } from "./tools.js";
import type { ReMeConfigInput, ReMeSettings } from "./types.js";

export const name = "reme-memory";
export const inject = ["agents", "sessions", "tools"];

export function apply(ctx: Context, input: ReMeConfigInput = {}): void {
  const base = resolveConfig(input);
  let settingsSource: () => ReMeSettings = () => settingsFrom(base);
  const current = () => mergeSettings(base, settingsSource());
  const client = new ReMeClient(current);
  const runtime = new ReMeRuntime(client, current, ctx.logger);
  installSettingsSection(
    ctx,
    REME_SETTINGS_NAMESPACE,
    SettingsConfig,
    settingsFrom(base),
    {
      setSource: (source) => {
        settingsSource = source;
      },
      onChange: () => {
        runtime.reconfigure();
      },
      validate: validateSettings,
    },
  );
  ctx.provide("remeMemory", runtime);
  void ctx.plugin(ReMeStatusGateway);
  ctx.effect(
    () => registerReMeTools(ctx, client, current),
    "remeMemory.tools()",
  );

  ctx.effect(() => {
    runtime.start();
    return () => runtime.disposeAll();
  }, "remeMemory.lifecycle()");

  ctx.on("agent/session-start", ({ agent }) => {
    const config = current();
    if (config.rootAgentsOnly && agent.session.header?.origin === "subagent")
      return;
    agent.ctx.effect(
      () => () => runtime.dispose(agent.session),
      "remeMemory.disposeSession()",
    );
    if (
      agent.status !== "idle" ||
      hasGuidance(agent.session, agent.inbox.nextStep)
    )
      return;
    agent.inject(
      createUserMessage({
        content: [{ type: "text", text: memoryGuidance(config.language) }],
        source: {
          kind: "plugin",
          plugin: REME_PLUGIN_SOURCE,
          form: "instructions",
        },
      }),
    );
  });

  ctx.on("session/event", (session, event) => {
    const config = current();
    if (config.rootAgentsOnly && session.header?.origin === "subagent") return;
    runtime.capture(session, event);
  });
}

export type { ReMeConfig, ReMeConfigInput, ReMeSettings } from "./types.js";
export type {
  ReMeRuntimeSnapshot,
  ReMeRuntimeTask,
  ReMeRuntimeTaskPhase,
} from "./runtime-status.js";
export { Config, REME_SETTINGS_NAMESPACE, SettingsConfig } from "./config.js";
