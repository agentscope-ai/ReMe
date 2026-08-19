import { createUserMessage } from "@deepseek-ai/dsh-llm";

import { ReMeClient } from "./client.js";
import { resolveConfig } from "./config.js";
import { hasGuidance, memoryGuidance, REME_PLUGIN_SOURCE } from "./guidance.js";
import { ReMeRuntime } from "./runtime.js";
import { registerReMeTools, type ToolRegistryContext } from "./tools.js";
import type { DshSession, LoggerLike, ReMeConfigInput, SessionEvent } from "./types.js";

type Cleanup = () => void | Promise<void>;

interface EffectContext {
  effect(execute: () => void | Cleanup, label?: string): unknown;
}

interface DshAgent {
  status: string;
  session: DshSession;
  ctx: EffectContext;
  inject(message: ReturnType<typeof createUserMessage>): unknown;
}

export interface DshPluginContext extends EffectContext, ToolRegistryContext {
  logger?: LoggerLike;
  provide(name: string, value: unknown): unknown;
  on(name: "agent/session-start", handler: (payload: { agent: DshAgent; source: string }) => void): unknown;
  on(name: "session/event", handler: (session: DshSession, event: SessionEvent) => void): unknown;
}

export const name = "reme-memory";
export const inject = ["agents", "sessions", "tools"];

export function apply(ctx: DshPluginContext, input: ReMeConfigInput = {}): void {
  const config = resolveConfig(input);
  const client = new ReMeClient(config);
  const runtime = new ReMeRuntime(client, config, ctx.logger);
  ctx.provide("remeMemory", runtime);
  registerReMeTools(ctx, client, config);

  ctx.effect(() => {
    runtime.start();
    return () => runtime.disposeAll();
  }, "remeMemory.lifecycle()");

  ctx.on("agent/session-start", ({ agent }) => {
    if (config.rootAgentsOnly && agent.session.header?.origin === "subagent") return;
    agent.ctx.effect(
      () => () => runtime.dispose(agent.session),
      "remeMemory.disposeSession()",
    );
    if (agent.status !== "idle" || hasGuidance(agent.session)) return;
    agent.inject(createUserMessage({
      content: [{ type: "text", text: memoryGuidance(config.language) }],
      source: { kind: "plugin", plugin: REME_PLUGIN_SOURCE, form: "instructions" },
    }));
  });

  ctx.on("session/event", (session, event) => {
    if (config.rootAgentsOnly && session.header?.origin === "subagent") return;
    runtime.capture(session, event);
  });
}

export type { ReMeConfig, ReMeConfigInput } from "./types.js";
