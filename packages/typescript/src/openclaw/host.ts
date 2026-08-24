/** Minimal OpenClaw API used by the ReMe adapter. */
export interface OpenClawPluginApi {
  pluginConfig?: Record<string, unknown>;
  logger: {
    info(message: string): void;
    warn(message: string): void;
    error(message: string): void;
  };
  registerTool(tool: OpenClawTool, options?: { name?: string }): void;
  on(name: "before_agent_start", handler: BeforeAgentStartHandler): void;
  on(name: "agent_end", handler: AgentEndHandler): void;
  registerService(service: {
    id: string;
    start(): void | Promise<void>;
    stop?(): void | Promise<void>;
  }): void;
}

export interface OpenClawPluginDefinition {
  id: string;
  name: string;
  description: string;
  kind: "memory";
  configSchema: {
    jsonSchema: Record<string, unknown>;
    parse(value: unknown): unknown;
  };
  register(api: OpenClawPluginApi): void | Promise<void>;
}

export interface OpenClawTool {
  name: string;
  label: string;
  description: string;
  parameters: object;
  execute(
    toolCallId: string,
    params: unknown,
  ): Promise<{
    content: Array<{ type: "text"; text: string }>;
    details: Record<string, unknown>;
  }>;
}

interface OpenClawAgentContext {
  agentId?: string;
  sessionId?: string;
  sessionKey?: string;
  trigger?: string;
}

type BeforeAgentStartHandler = (
  event: { prompt: string; messages?: unknown[] },
  context: OpenClawAgentContext,
) =>
  | Promise<{ prependContext?: string } | void>
  | { prependContext?: string }
  | void;

type AgentEndHandler = (
  event: { messages: unknown[]; success: boolean; error?: string },
  context: OpenClawAgentContext,
) => Promise<void> | void;
