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

export type StreamPayload = string | Record<string, unknown> | unknown[];

export interface StreamChunk {
  chunk_type: StreamChunkType | (string & {});
  chunk: StreamPayload;
  done: boolean;
  session_id?: string;
  block_id?: string;
  tool_call_id?: string;
  tool_call_name?: string;
  media_type?: string;
  input_tokens?: number;
  output_tokens?: number;
  metadata?: Record<string, unknown>;
}

export interface ContentBlock {
  id: string;
  type: "content";
  text: string;
}

export interface DetailBlock {
  id: string;
  type: "think" | "data" | "approval" | "usage" | "unknown";
  sourceType: string;
  payloads: StreamPayload[];
  status: "streaming" | "done" | "error";
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
  status: "calling" | "running" | "done" | "error";
  expanded: boolean;
  mediaType?: string;
  metadata?: Record<string, unknown>;
}

export interface ErrorBlock {
  id: string;
  type: "error";
  text: string;
}

export type ChatBlock = ContentBlock | DetailBlock | ToolBlock | ErrorBlock;

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  blocks?: ChatBlock[];
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
