import type {
  Activities,
  AgentActivity,
  AnimationIntent,
  AnimationManifest,
  Behavior,
  DesktopState,
} from "../shared/protocol.ts";

const terminal = (phase: AgentActivity["phase"]) =>
  ["completed", "failed", "interrupted"].includes(phase);
const clips = {
  thinking: "thinking",
  working: "thinking",
  remembering: "remembering",
  answering: "speaking",
  waiting: "listening",
  completed: "happy",
  failed: "confused",
  interrupted: "idle",
};
const labels = {
  thinking: "正在认真想…",
  working: "正在查找和处理…",
  remembering: "正在回忆…",
  answering: "正在回复…",
  waiting: "等你确认…",
  completed: "完成啦",
  failed: "这次没有完成",
  interrupted: "回复已停止",
};
type Owner =
  | "base"
  | "chat"
  | "background"
  | "reminder"
  | "terminal"
  | "preview"
  | "ambient";
type Input = Pick<
  DesktopState,
  "connection" | "chatFocused" | "pending" | "bubble" | "settings"
>;

/** One owner per animation. Temporary gestures always return to current activity. */
export class BehaviorController {
  value: Behavior = {
    animation: { id: 0, clip: "idle", mode: "loop" },
    label: "你的记忆，在这里",
  };
  dragging = false;
  private activities: Activities = {};
  private key = "idle";
  private owner: Owner = "base";
  private sequence = 0;
  private since = 0;
  private deadline = Infinity;
  private touched: number;
  private nextAmbient: number;
  private ambient: string | null = null;
  private lastAmbient = "";
  private preview: string | null = null;
  private cue: {
    key: string;
    clip: string;
    label: string;
    until: number;
  } | null = null;
  private seenTerminal = "";
  private reminder: { id: string; stage: "enter" | "hold" | "exit" } | null =
    null;

  constructor(
    private readonly manifest: AnimationManifest,
    now = Date.now(),
    private readonly random = Math.random,
  ) {
    this.touched = now;
    this.nextAmbient = this.nextIdle(now);
  }

  private nextIdle(now: number): number {
    return now + 15_000 + this.random() * 30_000;
  }

  get foregroundActive(): boolean {
    return Boolean(
      this.activities.chat && !terminal(this.activities.chat.phase),
    );
  }

  touch(now = Date.now()): void {
    this.touched = now;
    this.ambient = null;
    this.nextAmbient = this.nextIdle(now);
  }

  reset(now = Date.now()): void {
    this.activities = {};
    this.cue = null;
    this.preview = null;
    this.reminder = null;
    this.seenTerminal = "";
    this.dragging = false;
    this.touch(now);
  }

  observe(activities: Activities, live: boolean, now = Date.now()): void {
    this.activities = activities;
    const activity = activities.chat;
    if (!activity) {
      this.cue = null;
      return;
    }
    const key = `${activity.session_id}:${activity.turn_id}:${activity.phase}`;
    if (!terminal(activity.phase)) {
      this.cue = null;
      return;
    }
    if (key === this.seenTerminal) return;
    this.seenTerminal = key;
    if (live && activity.phase !== "interrupted") {
      const clip = clips[activity.phase];
      this.cue = {
        key,
        clip,
        label: labels[activity.phase],
        until: now + this.duration(clip) + 500,
      };
    }
  }

  play(clip: string, now = Date.now()): void {
    if (!this.manifest.animations.some((item) => item.id === clip))
      throw new Error("未知的角色动作");
    this.touch(now);
    this.preview = clip;
    // Explicit repeat requests are new playback instances.
    if (this.owner === "preview") this.key = "";
  }

  ended(id: number, now = Date.now()): void {
    if (id !== this.value.animation.id || this.value.animation.mode !== "once")
      return;
    if (this.owner === "reminder" && this.reminder) {
      this.reminder =
        this.reminder.stage === "enter"
          ? { ...this.reminder, stage: "hold" }
          : null;
    } else if (this.owner === "terminal") this.cue = null;
    else if (this.owner === "preview") this.preview = null;
    else if (this.owner === "ambient") {
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
    }
    this.deadline = Infinity;
  }

  private duration(clip: string, segment?: AnimationIntent["segment"]): number {
    const item = this.manifest.animations.find((value) => value.id === clip)!;
    const frames = segment
      ? item.segments![segment]
      : item.ticks.map((_, index) => index);
    return (
      (frames.reduce((sum, frame) => sum + item.ticks[frame], 0) /
        this.manifest.output_fps) *
      1000
    );
  }

  private choose(
    owner: Owner,
    key: string,
    clip: string,
    mode: AnimationIntent["mode"],
    label: string,
    now: number,
    segment?: AnimationIntent["segment"],
  ): Behavior {
    if (key !== this.key) {
      const smooth =
        owner === "chat" &&
        this.owner === "chat" &&
        mode === "loop" &&
        this.value.animation.mode === "loop" &&
        ["thinking", "remembering"].includes(clip) &&
        ["thinking", "remembering"].includes(this.value.animation.clip);
      if (smooth && now - this.since < 350) return this.value;
      this.key = key;
      this.owner = owner;
      this.since = now;
      this.deadline =
        mode === "once" ? now + this.duration(clip, segment) + 1000 : Infinity;
      this.value = {
        animation: {
          id: ++this.sequence,
          clip,
          mode,
          ...(segment ? { segment } : {}),
        },
        label,
      };
    } else if (label !== this.value.label)
      this.value = { ...this.value, label };
    return this.value;
  }

  resolve(input: Input, visible: boolean, now = Date.now()): Behavior {
    if (now >= this.deadline) this.ended(this.value.animation.id, now);
    if (this.cue && now >= this.cue.until) this.cue = null;
    const select = (
      owner: Owner,
      key: string,
      clip: string,
      mode: AnimationIntent["mode"],
      label: string,
      segment?: AnimationIntent["segment"],
    ) => this.choose(owner, key, clip, mode, label, now, segment);
    if (this.dragging)
      return select("base", "dragging", "dragged", "loop", "轻轻挪一挪");
    if (!visible || input.connection !== "connected") {
      this.reminder = null;
      return select("base", "offline", "idle", "hold", "等待连接 ReMe");
    }
    const activity = this.activities.chat;
    const active = activity && !terminal(activity.phase);
    const starting =
      input.pending?.streaming &&
      (!activity ||
        (terminal(activity.phase) &&
          activity.turn_id !== input.pending.turnId));
    if (active || starting) {
      this.preview = null;
      this.ambient = null;
      this.reminder = null;
      this.touched = now;
      this.nextAmbient = this.nextIdle(now);
      const phase = active ? activity.phase : "thinking";
      return select(
        "chat",
        `${activity?.turn_id ?? input.pending?.id}:${phase}`,
        clips[phase],
        "loop",
        labels[phase],
      );
    }
    if (input.bubble && input.bubble.id !== this.reminder?.id)
      this.reminder = { id: input.bubble.id, stage: "enter" };
    else if (!input.bubble && this.reminder && this.reminder.stage !== "exit")
      this.reminder.stage = "exit";
    if (this.reminder) {
      this.touched = now;
      this.nextAmbient = this.nextIdle(now);
      const { id, stage } = this.reminder;
      return select(
        "reminder",
        `reminder:${id}:${stage}`,
        "reminding",
        stage === "hold" ? "hold" : "once",
        "想到一件值得告诉你的事",
        stage,
      );
    }
    if (this.cue)
      return select(
        "terminal",
        this.cue.key,
        this.cue.clip,
        "once",
        this.cue.label,
      );
    if (this.preview)
      return select(
        "preview",
        `preview:${this.preview}`,
        this.preview,
        "once",
        "你的记忆，在这里",
      );
    if (input.chatFocused && now - this.touched < 15_000)
      return select("base", "listening", "listening", "loop", "我在听，慢慢说");
    const background = this.activities.proactive;
    const quiet = input.settings.quietUntil > now;
    if (input.chatFocused || quiet) this.ambient = null;
    if (
      background &&
      !terminal(background.phase) &&
      !input.chatFocused &&
      !quiet &&
      now - this.touched < 300_000
    )
      return select(
        "background",
        `background:${background.turn_id}:${background.phase}`,
        clips[background.phase],
        "loop",
        "在整理一些信息…",
      );
    if (now - this.touched >= 300_000)
      return select("base", "sleeping", "sleeping", "loop", "休息一会儿");
    if (
      !input.chatFocused &&
      !quiet &&
      !this.ambient &&
      now >= this.nextAmbient
    ) {
      const pool = this.manifest.ambient.filter(
        (clip) => clip !== this.lastAmbient,
      );
      this.ambient = pool[Math.floor(this.random() * pool.length)];
      this.lastAmbient = this.ambient;
    }
    if (this.ambient && !input.chatFocused && !quiet)
      return select(
        "ambient",
        `ambient:${this.ambient}`,
        this.ambient,
        "once",
        "安静陪着你",
      );
    return select("base", "idle", "idle", "loop", "你的记忆，在这里");
  }
}
