import {
  defaultLanguage,
  translate,
  translator,
  UiError,
} from "../shared/i18n.ts";
import type { Language, TranslationKey } from "../shared/i18n.ts";
import type {
  Activities,
  AgentActivity,
  AnimationIntent,
  AnimationManifest,
  Behavior,
  DesktopState,
  PlaybackStatus,
  StatusNotice,
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
    label: translate(defaultLanguage, "idle"),
    notice: null,
    error: "",
  };
  dragging = false;
  private activities: Activities = {};
  private key = "idle";
  private owner: Owner = "base";
  private sequence = 0;
  private since = 0;
  private deadline = Infinity;
  private started = false;
  private readonly unavailable = new Set<string>();
  private language: Language = defaultLanguage;
  private touched: number;
  private nextAmbient: number;
  private ambient: string | null = null;
  private lastAmbient = "";
  private preview: string | null = null;
  private cue: {
    key: string;
    clip: string;
    label: TranslationKey;
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
        label: activity.phase,
        until: now + this.duration(clip) + 500,
      };
    }
  }

  play(clip: string, now = Date.now()): void {
    if (!this.manifest.animations.some((item) => item.id === clip))
      throw new UiError("unknownAnimation");
    this.touch(now);
    this.preview = clip;
    // Explicit repeat requests are new playback instances.
    if (this.owner === "preview") this.key = "";
  }

  report(id: number, status: PlaybackStatus, now = Date.now()): void {
    if (id !== this.value.animation.id) return;
    if (status === "started") {
      this.started = true;
      this.deadline =
        this.value.animation.mode === "once"
          ? now +
            this.duration(
              this.value.animation.clip,
              this.value.animation.segment,
            ) +
            5000
          : Infinity;
    } else if (status === "ended") this.ended(id, now);
    else {
      this.deadline = Infinity;
      if (this.unavailable.has(this.value.animation.clip)) return;
      this.unavailable.add(this.value.animation.clip);
      this.key = "";
      this.preview = null;
      this.ambient = null;
    }
  }

  private ended(id: number, now: number): void {
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
    notice: StatusNotice | null = null,
  ): Behavior {
    const error = this.unavailable.size
      ? translate(this.language, "animationUnavailable")
      : "";
    if (this.unavailable.has(clip)) {
      clip = "idle";
      mode = "hold";
      segment = undefined;
    }
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
      this.started = false;
      this.deadline = now + 5000;
      this.value = {
        animation: {
          id: ++this.sequence,
          clip,
          mode,
          ...(segment ? { segment } : {}),
        },
        label,
        notice,
        error,
      };
    } else if (
      label !== this.value.label ||
      error !== this.value.error ||
      JSON.stringify(notice) !== JSON.stringify(this.value.notice)
    )
      this.value = { ...this.value, label, notice, error };
    return this.value;
  }

  resolve(input: Input, visible: boolean, now = Date.now()): Behavior {
    this.language = input.settings.language;
    const t = translator(this.language);
    if (now >= this.deadline) {
      if (this.started) this.ended(this.value.animation.id, now);
      else this.report(this.value.animation.id, "failed", now);
    }
    if (this.cue && now >= this.cue.until) this.cue = null;
    const select = (
      owner: Owner,
      key: string,
      clip: string,
      mode: AnimationIntent["mode"],
      label: string,
      segment?: AnimationIntent["segment"],
      notice?: StatusNotice,
    ) => this.choose(owner, key, clip, mode, label, now, segment, notice);
    const notice = (
      id: string,
      text: string,
      tone: StatusNotice["tone"] = "working",
    ): StatusNotice => ({ kind: "status", id, text, tone });
    if (this.dragging)
      return select("base", "dragging", "dragged", "loop", t("dragging"));
    if (!visible || input.connection !== "connected") {
      this.reminder = null;
      return select("base", "offline", "idle", "hold", t("waitingConnection"));
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
        t(phase),
        undefined,
        notice(
          `chat:${activity?.turn_id ?? input.pending?.id}:${phase}`,
          t(phase),
          phase === "waiting" ? "waiting" : "working",
        ),
      );
    }
    if (input.bubble && input.bubble.id !== this.reminder?.id)
      this.reminder = { id: input.bubble.id, stage: "enter" };
    else if (!input.bubble && this.reminder && this.reminder.stage !== "exit")
      this.reminder.stage = "exit";
    if (this.reminder) {
      this.touched = now;
      this.nextAmbient = this.nextIdle(now);
      if (this.unavailable.has("reminding")) {
        this.reminder = null;
        return select(
          "base",
          "reminder-unavailable",
          "idle",
          "hold",
          t("reminderStatus"),
        );
      }
      const { id, stage } = this.reminder;
      return select(
        "reminder",
        `reminder:${id}:${stage}`,
        "reminding",
        stage === "hold" ? "hold" : "once",
        t("reminderStatus"),
        stage,
      );
    }
    if (this.cue)
      return select(
        "terminal",
        this.cue.key,
        this.cue.clip,
        "once",
        t(this.cue.label),
        undefined,
        notice(
          this.cue.key,
          t(this.cue.label),
          this.cue.clip === "happy" ? "success" : "error",
        ),
      );
    if (this.preview)
      return select(
        "preview",
        `preview:${this.preview}`,
        this.preview,
        "once",
        t("idle"),
      );
    if (input.chatFocused && now - this.touched < 15_000)
      return select("base", "listening", "listening", "loop", t("listening"));
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
        t("background"),
        undefined,
        notice(
          `background:${background.turn_id}:${background.phase}`,
          t("background"),
        ),
      );
    if (now - this.touched >= 300_000)
      return select("base", "sleeping", "sleeping", "loop", t("sleeping"));
    if (
      !input.chatFocused &&
      !quiet &&
      !this.ambient &&
      now >= this.nextAmbient
    ) {
      const pool = this.manifest.ambient.filter(
        (clip) => clip !== this.lastAmbient && !this.unavailable.has(clip),
      );
      if (pool.length) {
        this.ambient = pool[Math.floor(this.random() * pool.length)];
        this.lastAmbient = this.ambient;
      }
    }
    if (this.ambient && !input.chatFocused && !quiet)
      return select(
        "ambient",
        `ambient:${this.ambient}`,
        this.ambient,
        "once",
        t("ambient"),
      );
    return select("base", "idle", "idle", "loop", t("idle"));
  }
}
