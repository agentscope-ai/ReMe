export interface ReMeResponse<TAnswer = unknown> {
  answer: TAnswer;
  success: boolean;
  metadata: Record<string, unknown>;
}

export interface AppConfig {
  app_name: string;
  workspace_dir: string;
  daily_dir: string;
  digest_dir: string;
  resource_dir: string;
  [key: string]: unknown;
}

export interface ReMeComponentHealth {
  is_started?: boolean;
  is_healthy?: boolean | null;
  model_name?: string | null;
  dimensions?: number | null;
  cache_size?: number;
  n_nodes?: number;
  n_edges?: number;
  n_virtual?: number;
  n_pending?: number;
  n_chunks?: number;
  n_chunks_with_embedding?: number;
  n_docs?: number | null;
  vocab_size?: number;
  memory?: string;
}

export interface ReMeHealth {
  version: string;
  healthy: boolean;
  components: Record<string, Record<string, ReMeComponentHealth>>;
}

export interface FileStat {
  path: string;
  exists: boolean;
  type: "file" | "dir";
  mtime?: string;
  size?: number;
  mime?: string;
  frontmatter?: Record<string, unknown>;
}

export interface TreeNode {
  name: string;
  path: string;
  type: "directory" | "file";
  children: TreeNode[];
}

export type WorkspaceSource = "workspace" | "daily" | "digest";
export type MemoryGraphRoot = "wiki" | "personal" | "procedure";

export interface GraphSnapshotNode {
  id: string;
  path: string;
  name: string;
  description: string;
  indexed: boolean;
  virtual: boolean;
}

export interface GraphSnapshotEdge {
  source: string;
  target: string;
  target_anchor: string | null;
}

export interface GraphSnapshot {
  version: 1;
  nodes: GraphSnapshotNode[];
  edges: GraphSnapshotEdge[];
}

export type StreamChunkType =
  | "reply_start"
  | "reply_end"
  | "think"
  | "content"
  | "data"
  | "tool_call"
  | "tool_result"
  | "approval"
  | "usage"
  | "error"
  | "done";

export type StreamPayload =
  | string
  | number
  | boolean
  | Record<string, unknown>
  | unknown[]
  | null;
export type StreamStatus =
  | "running"
  | "completed"
  | "failed"
  | "interrupted"
  | "waiting";

export interface StreamChunk {
  chunk_type: StreamChunkType | (string & {});
  event?: "start" | "delta" | "end";
  chunk: StreamPayload;
  done: boolean;
  status?: StreamStatus | null;
  session_id?: string | null;
  turn_id?: string | null;
  message_id?: string | null;
  block_id?: string | null;
  tool_call_id?: string | null;
  tool_call_name?: string | null;
  media_type?: string | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  metadata?: Record<string, unknown>;
}

export interface AgentReply {
  session_id: string;
  result: string;
  structured_output: unknown;
  usage: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  } | null;
  status: Exclude<StreamStatus, "running">;
  error: string | null;
  raw: Record<string, unknown>;
  events: StreamChunk[];
}

export interface ContentBlock {
  id: string;
  type: "content";
  text: string;
  metadata?: Record<string, unknown>;
}

export interface DetailBlock {
  id: string;
  type: "think" | "data" | "approval" | "usage" | "unknown";
  sourceType: string;
  payloads: StreamPayload[];
  status: "streaming" | "done" | "error" | "interrupted" | "waiting";
  expanded: boolean;
  mediaType?: string;
  inputTokens?: number;
  outputTokens?: number;
  metadata?: Record<string, unknown>;
}

export interface ToolBlock {
  id: string;
  type: "tool";
  name: string;
  callPayloads: StreamPayload[];
  resultPayloads: StreamPayload[];
  status: "calling" | "running" | "done" | "error" | "interrupted" | "waiting";
  expanded: boolean;
  mediaType?: string;
  metadata?: Record<string, unknown>;
}

export interface ErrorBlock {
  id: string;
  type: "error";
  text: string;
  metadata?: Record<string, unknown>;
}

export type ChatBlock = ContentBlock | DetailBlock | ToolBlock | ErrorBlock;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  blocks?: ChatBlock[];
  events?: StreamChunk[];
  reply?: Omit<AgentReply, "events">;
  status?: StreamStatus;
  turnId?: string;
  source?: "chat" | "proactive";
  createdAt?: string | null;
  localOnly?: boolean;
}

export interface SessionMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  source: "chat" | "proactive";
  created_at: string | null;
  turn_id: string | null;
  completed: boolean;
  chunks: StreamChunk[];
}

export interface SessionSnapshot {
  session_id: string;
  state: {
    session_id: string;
    status: "active" | "idle" | "not_loaded" | "error";
    active_flags: string[];
  };
  messages: SessionMessage[];
}

export type WorkspaceTab =
  | {
      id: string;
      type: "markdown";
      title: string;
      path: string;
      content: string;
      savedContent: string;
      mtime?: string;
      loading?: boolean;
      error?: string;
    }
  | {
      id: string;
      type: "agent";
      title: string;
      sessionId?: string;
      messages: ChatMessage[];
      streaming?: boolean;
    }
  | {
      id: string;
      type: "graph";
      title: string;
      root: MemoryGraphRoot;
    };
