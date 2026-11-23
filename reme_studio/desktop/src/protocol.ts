import type { ErrorData } from "../../client/src/i18n.ts";
import type {
  ItemTarget,
  PlaybackStatus,
  StudioAction,
  StudioApi,
  StudioState,
} from "../../client/src/protocol.ts";

export type DesktopPatch = Partial<StudioState> & { revision: number };

export interface AnimationClip {
  id: string;
  ticks: number[];
  frame_files: string[];
  segments?: Record<"enter" | "hold" | "exit", number[]>;
}

export interface AnimationManifest {
  version: 2;
  ambient: string[];
  ambient_weights?: Record<string, number>;
  canvas: [number, number];
  output_fps: number;
  animations: AnimationClip[];
}

export interface WindowLayout {
  visible: boolean;
  body: { left: number; top: number; size: number };
  bubble: {
    left: number;
    top: number;
    width: number;
    height: number;
    side: "top" | "bottom" | "left" | "right";
    tip: number;
  };
}

export interface Bootstrap {
  state: StudioState;
  manifest: AnimationManifest;
  layout: WindowLayout;
  platform: string;
  version: string;
}

export type DesktopAction =
  | StudioAction
  | { type: "playback.report"; id: number; status: PlaybackStatus }
  | {
      type: "bubble.shown";
      sessionId: string;
      id: string;
      presentationId: number;
    }
  | { type: "companion.menu" }
  | { type: "companion.menu.close" }
  | { type: "settings.open" }
  | { type: "animation.play"; clip: string }
  | { type: "app.quit" }
  | { type: "chat.open"; target?: ItemTarget }
  | { type: "chat.close" }
  | { type: "window.minimize" }
  | { type: "window.maximize" }
  | { type: "companion.reset" }
  | { type: "companion.visibility"; visible: boolean }
  | { type: "bubble.dismiss"; id: string }
  | { type: "bubble.snooze"; id: string }
  | { type: "bubble.pause"; id: string; paused: boolean };

export type UpdateCommand = "check" | "download" | "install";

export type UpdateState =
  | { status: "manual" | "idle" | "checking" | "current" | "error" }
  | { status: "available" | "ready" | "installing"; version: string }
  | { status: "downloading"; version: string; percent: number };

export interface DesktopApi extends StudioApi {
  bootstrap(): Promise<Bootstrap>;
  action(action: DesktopAction): Promise<ErrorData | null>;
  onState(callback: (state: DesktopPatch) => void): () => void;
  onLayout(callback: (layout: WindowLayout) => void): () => void;
  update(command: UpdateCommand): Promise<UpdateState>;
  onUpdate(callback: (state: UpdateState) => void): () => void;
  pointer(interactive: boolean): void;
  drag(phase: "start" | "move" | "end"): void;
}

export const channels = {
  bootstrap: "reme:bootstrap",
  action: "reme:action",
  state: "reme:state",
  layout: "reme:layout",
  pointer: "reme:pointer",
  drag: "reme:drag",
  command: "reme:command",
  transcribe: "reme:transcribe",
  workspace: "reme:workspace",
  update: "reme:update",
  updateState: "reme:update-state",
} as const;
