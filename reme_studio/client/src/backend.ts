export const reasoningEfforts = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
] as const;
export type ReasoningEffort = (typeof reasoningEfforts)[number];
export type Harness = "codex" | "claude_code";

export function harnessEfforts(
  harness: Harness | null,
): readonly ReasoningEffort[] {
  return harness === "claude_code"
    ? reasoningEfforts.filter(
        (effort) => effort !== "none" && effort !== "minimal",
      )
    : reasoningEfforts;
}

export interface BackendValues {
  workspace_dir: string | null;
  timezone: string | null;
  language: string;
  harness: Harness | null;
  base_url: string | null;
  model: string | null;
  effort: ReasoningEffort | null;
  asr_enabled: boolean;
  tts_enabled: boolean;
  speech_base_url: string;
  asr_model: string;
  tts_model: string;
  tts_voice: string;
}
export interface BackendSettings extends BackendValues {
  revision: number;
  initialized: boolean;
  settings_error: string | null;
  model_editable: boolean;
  available_harnesses: Harness[];
  api_key_set: boolean;
  dashscope_api_key_set: boolean;
  dashscope_editable: boolean;
  asr_available: boolean;
  tts_available: boolean;
  missing_fields: string[];
  defaults: Pick<
    BackendValues,
    "speech_base_url" | "asr_model" | "tts_model" | "tts_voice"
  >;
}
export type BackendPatch = {
  revision: number;
  api_key?: string;
  dashscope_api_key?: string;
} & Partial<BackendValues>;
export interface BackendOperation {
  id: string;
  phase: "accepted" | "restarting" | "succeeded" | "failed";
  error: string | null;
}
export interface BackendStatus {
  ready: boolean;
  needs_setup: boolean;
  missing_fields: string[];
  revision: number;
  error: string | null;
  operation: BackendOperation | null;
}
export type BackendRequest = {
  type: "backend";
  credentials?: { endpoint: string; token: string };
} & (
  | { action: "config" | "restart" }
  | { action: "status"; operationId?: string }
  | { action: "apply" | "test_model"; settings: BackendPatch }
);
export type NavigationGuard = (next: () => void) => void;
