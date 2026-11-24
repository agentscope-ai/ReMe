import { contract } from "./contract-fixture.ts";
import type {
  AgentMessage,
  AgentView,
  AgentSnapshot,
  ClientInfo,
} from "../src/wire.ts";

export function agent(
  name = "Remy",
  sessionId: string | null = "session",
): AgentView {
  return {
    ...contract.agent,
    agent_name: name,
    session_id: sessionId,
    agent_type: name === "Remy" ? "chat" : "event",
    directory: `reme_agents/${name}`,
    memory_path: name === "Remy" ? `reme_agents/${name}/MEMORY.md` : null,
    task_path: name === "Remy" ? null : `reme_agents/${name}/TASK.md`,
  };
}

export function message(fields: Partial<AgentMessage> = {}): AgentMessage {
  return { ...contract.messages[0], items: [], ...fields };
}

export function snapshot(
  messages: AgentMessage[] = [],
  sessionId = "session",
  name = "Remy",
): AgentSnapshot {
  return {
    type: "snapshot",
    agent: agent(name, sessionId),
    messages,
    activities: {},
    agents: null,
  };
}

export const clientInfo: ClientInfo = {
  workspace_dir: "/test-workspace",
  timezone: "Asia/Shanghai",
  resource_dir: "resource",
  daily_dir: "daily",
  digest_dir: "digest",
  tag_key: "memory_tags",
  jobs: [
    "watch_agent_session",
    "ask_agent",
    "read_agent_session",
    "create_chat_session",
    "list_task_reminders",
    "read_agent_reminder",
    "list_agent_inbox",
  ],
  asr: false,
  tts: false,
  speech_revision: "fixture",
};
