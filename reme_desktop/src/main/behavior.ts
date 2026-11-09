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
  Settings,
  TaskActivities,
} from "../shared/protocol.ts";

import { terminal } from "../shared/messages.ts";
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
const ambientIntervals = {
  calm: [40_000, 80_000],
  normal: [18_000, 40_000],
  lively: [8_000, 18_000],
};
const ambientCooldowns: Record<string, number> = {
  petting_fish: 60_000,
  listening: 30_000,
  remembering: 75_000,
  thinking: 90_000,
};
type Owner =
  | "base"
  | "chat"
  | "speech"
  | "background"
  | "reminder"
  | "terminal"
  | "preview"
  | "ambient";
type Input = Pick<
  DesktopState,
  "connection" | "chatFocused" | "pending" | "settings" | "speech"
> & { reminderId: string | null };

/** One owner per animation. Temporary gestures always return to current activity. */
export class BehaviorController {
  value: Behavior = {
    animation: { id: 0, clip: "idle", mode: "loop" },
    label: translate(defaultLanguage, "idle"),
    error: "",
  };
  dragging = false;
  private activities: Activities = {};
  private taskActivities: TaskActivities = {};
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
  private recentAmbient: string[] = [];
  private readonly ambientPlayed = new Map<string, number>();
  private activityLevel: Settings["activityLevel"] = "normal";
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
    this.value.animation.startedAt = now;
  }

  private nextIdle(now: number): number {
    const [min, max] = ambientIntervals[this.activityLevel];
    const slower = now - this.touched >= 120_000 ? 1.5 : 1;
    return now + (min + this.random() * (max - min)) * slower;
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
    this.taskActivities = {};
    this.cue = null;
    this.preview = null;
    this.reminder = null;
    this.seenTerminal = "";
    this.dragging = false;
    this.recentAmbient = [];
    this.ambientPlayed.clear();
    this.touch(now);
  }

  observe(
    activities: Activities,
    live: boolean,
    now = Date.now(),
    tasks: TaskActivities = {},
  ): void {
    this.activities = activities;
    this.taskActivities = tasks;
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
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
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
      // A phase or owner change can keep the same pose and playback clock.
      if (
        clip === this.value.animation.clip &&
        mode === "loop" &&
        this.value.animation.mode === "loop" &&
        segment === this.value.animation.segment
      ) {
        this.key = key;
        this.owner = owner;
        this.value = { ...this.value, label, error };
        return this.value;
      }
      const smooth =
        owner === "chat" &&
        this.owner === "chat" &&
        mode === "loop" &&
        this.value.animation.mode === "loop" &&
        ["thinking", "remembering"].includes(clip) &&
        ["thinking", "remembering"].includes(this.value.animation.clip);
      if (smooth && now - this.since < 350) {
        this.value = { ...this.value, label, error };
        return this.value;
      }
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
          startedAt: now,
          ...(segment ? { segment } : {}),
        },
        label,
        error,
      };
    } else if (label !== this.value.label || error !== this.value.error)
      this.value = { ...this.value, label, error };
    return this.value;
  }

  resolve(input: Input, visible: boolean, now = Date.now()): Behavior {
    this.language = input.settings.language;
    const level = input.settings.activityLevel;
    if (level !== this.activityLevel) {
      this.activityLevel = level;
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
    }
    const t = translator(this.language);
    if (!visible) {
      this.reminder = null;
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
      const behavior = this.choose(
        "base",
        "offline",
        "idle",
        "hold",
        t("waitingConnection"),
        now,
      );
      this.deadline = Infinity;
      return behavior;
    }
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
    ) => this.choose(owner, key, clip, mode, label, now, segment);
    if (this.dragging)
      return select("base", "dragging", "dragged", "loop", t("dragging"));
    if (input.connection !== "connected") {
      this.reminder = null;
      return select("base", "offline", "idle", "hold", t("waitingConnection"));
    }
    const speech = input.speech;
    if (speech?.capture || speech?.playback?.phase === "playing") {
      this.touch(now);
      const capture = speech.capture;
      const label = capture
        ? t(
            capture.phase === "transcribing"
              ? "speechTranscribing"
              : capture.phase === "starting"
                ? "speechOpeningMic"
                : "speechListening",
          )
        : t("speechPlaying");
      return select(
        "speech",
        `speech:${capture?.id ?? speech.playback!.id}:${capture?.phase ?? "playing"}`,
        capture
          ? capture.phase === "transcribing"
            ? "thinking"
            : "listening"
          : "speaking",
        "loop",
        label,
      );
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
      );
    }
    if (input.reminderId && input.reminderId !== this.reminder?.id)
      this.reminder = { id: input.reminderId, stage: "enter" };
    else if (
      !input.reminderId &&
      this.reminder &&
      this.reminder.stage !== "exit"
    )
      this.reminder.stage = "exit";
    if (this.reminder) {
      this.touched = now;
      this.ambient = null;
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
    const task = Object.values(this.taskActivities).find(
      (item) => !terminal(item.phase),
    );
    const proactive = this.activities.proactive;
    const background =
      task ?? (proactive && !terminal(proactive.phase) ? proactive : undefined);
    const quiet = input.settings.quietUntil > now;
    if (input.chatFocused || quiet) {
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
    }
    if (background) {
      this.touched = now;
      this.ambient = null;
      this.nextAmbient = this.nextIdle(now);
    }
    if (background && !input.chatFocused && !quiet) {
      const label = task
        ? t("taskBackground", { task: task.task_id! })
        : t("background");
      return select(
        "background",
        `background:${background.run_id ?? background.turn_id}:${background.phase}`,
        background.phase === "answering"
          ? "remembering"
          : clips[background.phase],
        "loop",
        label,
      );
    }
    const sleepMinutes = input.settings.sleepMinutes;
    if (sleepMinutes > 0 && now - this.touched >= sleepMinutes * 60_000)
      return select("base", "sleeping", "sleeping", "loop", t("sleeping"));
    if (
      !input.chatFocused &&
      !quiet &&
      !this.ambient &&
      now >= this.nextAmbient
    ) {
      const pool = this.manifest.ambient.filter(
        (clip) =>
          clip !== this.recentAmbient.at(-1) &&
          now - (this.ambientPlayed.get(clip) ?? -Infinity) >=
            (ambientCooldowns[clip] ?? 30_000) &&
          !this.unavailable.has(clip),
      );
      if (pool.length) {
        const weight = (clip: string) =>
          (this.manifest.ambient_weights?.[clip] ?? 1) *
          (this.recentAmbient.includes(clip) ? 0.35 : 1);
        let choice =
          this.random() * pool.reduce((sum, clip) => sum + weight(clip), 0);
        this.ambient = pool[pool.length - 1];
        for (const clip of pool) {
          choice -= weight(clip);
          if (choice < 0) {
            this.ambient = clip;
            break;
          }
        }
        this.ambientPlayed.set(this.ambient, now);
        this.recentAmbient = [...this.recentAmbient.slice(-1), this.ambient];
      } else this.nextAmbient = this.nextIdle(now);
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
