import { errorText, languages, translate, UiError } from "../shared/i18n.ts";
import { createHash, randomUUID } from "node:crypto";
import { BubbleQueue } from "./bubbles.ts";
import { BehaviorController } from "./behavior.ts";
import { NotificationPolicy } from "./notifications.ts";
import { emptySpeech, SpeechController } from "./speech.ts";
import type { SystemNotice } from "./notifications.ts";
import { delay, ReMeClient } from "./reme-client.ts";
import { SessionModel } from "./session-model.ts";
import { normalizeEndpoint, Storage } from "./storage.ts";
import { reduceReply } from "../shared/stream.ts";
import type { ReplyProgress } from "../shared/stream.ts";
import type {
  DesktopState,
  Presentation,
  Receipt,
  SessionMessage,
  SessionSnapshot,
  AnimationManifest,
  PlaybackStatus,
  SettingsUpdate,
  DesktopCommand,
} from "../shared/protocol.ts";

export class DesktopController {
  readonly model = new SessionModel();
  readonly bubbles: BubbleQueue;
  readonly state: DesktopState;
  readonly speech: SpeechController;
  private readonly behavior: BehaviorController;
  private scope = "";
  private lifecycle: AbortController | null = null;
  private stream: AbortController | null = null;
  private running: Promise<void> = Promise.resolve();
  private clock: ReturnType<typeof setInterval> | undefined;
  private ready = false;
  private companionVisible = true;
  private companionReady = false;
  private readonly notificationPolicy = new NotificationPolicy();
  private readonly notified = new Set<string>();
  private generation = 0;
  private endpoint: string;
  private settingsWrites: Promise<void> = Promise.resolve();
  private closed = false;
  private flushedAt = 0;
  private connectionIssue: unknown = null;
  private pendingIssue: unknown = null;

  constructor(
    readonly storage: Storage,
    private readonly publish: (state: DesktopState) => void,
    manifest: AnimationManifest,
    private readonly notify: (
      notice: SystemNotice,
      shown: () => void,
    ) => void = () => {},
    command: (command: DesktopCommand) => void = () => {},
  ) {
    this.behavior = new BehaviorController(manifest);
    this.endpoint = storage.settings.endpoint;
    this.state = {
      revision: 0,
      connection: "connecting",
      connectionError: "",
      settings: storage.settings,
      workspace: "",
      sessionId: null,
      status: null,
      messages: [],
      pending: null,
      bubble: null,
      surface: null,
      unread: 0,
      chatVisible: false,
      chatFocused: false,
      focusMessageId: null,
      behavior: this.behavior.value,
      speech: emptySpeech(),
    };
    this.bubbles = new BubbleQueue(10 * 60_000, (id) => {
      const receipt = this.presentation()?.receipts[id];
      if (receipt) {
        receipt.suppressed = true;
        this.storage.changed();
      }
    });
    this.speech = new SpeechController(
      () => ({
        sessionId: this.model.sessionId,
        connected: this.ready && !this.closed,
        chatVisible: this.state.chatVisible,
        visible: this.state.chatVisible || this.companionVisible,
        settings: this.state.settings,
      }),
      () => this.client(),
      (speech) => {
        this.state.speech = speech;
        this.emit();
      },
      command,
    );
  }

  start(): void {
    this.restart();
    this.clock = setInterval(() => {
      this.tick();
      if (Date.now() - this.flushedAt < 1000) return;
      this.flushedAt = Date.now();
      void this.storage.flush().catch((error) => {
        this.connectionIssue = new UiError("saveSettingsFailed", {
          error: errorText(error, this.state.settings.language),
        });
        this.emit();
      });
    }, 100);
  }

  private restart(): void {
    this.speech.reset();
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    this.endpoint = this.storage.settings.endpoint;
    this.ready = false;
    this.model.clear();
    this.behavior.reset();
    this.notificationPolicy.reset();
    this.notified.clear();
    this.bubbles.clear();
    this.state.bubble = null;
    this.state.pending = null;
    this.pendingIssue = null;
    this.state.workspace = "";
    this.scope = "";
    this.state.connection = "connecting";
    this.connectionIssue = null;
    this.lifecycle = new AbortController();
    const previous = this.running;
    const generation = this.generation;
    const signal = this.lifecycle.signal;
    this.running = previous
      .catch(() => {})
      .then(() => this.connect(generation, signal));
    this.emit();
  }

  private client(): ReMeClient {
    return new ReMeClient(this.endpoint);
  }

  private async connect(
    generation: number,
    signal: AbortSignal,
  ): Promise<void> {
    let attempt = 0;
    while (!signal.aborted) {
      try {
        const client = this.client();
        const config = await client.call<{
          workspace_dir: string;
          jobs: Record<
            string,
            { backend: string; enabled?: boolean; enable_serve?: boolean }
          >;
          components?: {
            tts?: Record<
              string,
              {
                model?: string;
                voice?: string;
                audio_format?: string;
                sample_rate?: number;
                parameters?: Record<string, unknown>;
              }
            >;
          };
        }>("app_config", {}, signal);
        if (signal.aborted) return;
        for (const job of [
          "pa_events",
          "pa_chat",
          "pa_session",
          "pa_new_session",
          "session_read",
        ]) {
          if (
            !config.jobs?.[job] ||
            config.jobs[job].enabled === false ||
            config.jobs[job].enable_serve === false
          ) {
            throw new UiError("missingJob", { job });
          }
        }
        if (typeof config.workspace_dir !== "string")
          throw new UiError("missingWorkspace");
        const available = (name: string) =>
          Boolean(
            config.jobs?.[name] &&
            config.jobs[name].enabled !== false &&
            config.jobs[name].enable_serve !== false,
          );
        const tts = config.components?.tts?.default;
        this.speech.configure(
          available("asr_demo"),
          available("tts_demo"),
          JSON.stringify([
            client.endpoint,
            config.workspace_dir,
            tts?.model,
            tts?.voice,
            tts?.audio_format,
            tts?.sample_rate,
            tts?.parameters,
          ]),
        );
        const scope = createHash("sha256")
          .update(client.endpoint + "\n" + config.workspace_dir)
          .digest("hex");
        if (scope !== this.scope) {
          this.model.clear();
          this.bubbles.clear();
          this.scope = scope;
        }
        this.state.workspace = config.workspace_dir;
        const current = await client.call<{ session_id: string | null }>(
          "pa_session",
          {},
          signal,
        );
        if (!current.session_id)
          await client.call("pa_new_session", {}, signal);
        if (signal.aborted) return;
        await client.stream(
          "pa_events",
          this.model.cursor ? { cursor: this.model.cursor } : {},
          signal,
          (chunk) => {
            if (signal.aborted || generation !== this.generation) return;
            if (chunk.chunk_type === "error") {
              if (typeof chunk.chunk === "string") throw new Error(chunk.chunk);
              throw new UiError("subscriptionError");
            }
            if (chunk.chunk_type !== "data" || !chunk.chunk) return;
            const previous = this.model.sessionId;
            let event;
            try {
              event = this.model.apply(chunk.chunk);
            } catch (error) {
              this.model.cursor = undefined;
              throw error;
            }
            if (
              event.type === "heartbeat" &&
              this.state.connection === "connected"
            )
              return;
            if (previous !== this.model.sessionId) {
              this.speech.reset();
              this.behavior.reset();
              this.notificationPolicy.reset();
              this.notified.clear();
              this.bubbles.clear();
              this.state.focusMessageId = null;
            }
            if (event.type === "snapshot" || event.type === "status_changed") {
              this.behavior.observe(
                this.model.activities,
                event.type !== "snapshot",
              );
            }
            this.ready = !this.model.waitingForSnapshot;
            this.state.connection = this.ready ? "connected" : "reconnecting";
            this.connectionIssue = event.error ?? null;
            if (event.type === "snapshot" || event.type === "status_changed") {
              const notice = this.notificationPolicy.observe(
                this.model.activities.chat,
                event.type !== "snapshot",
                this.notificationEligible(),
                Date.now(),
                this.state.settings.language,
              );
              if (notice && this.model.sessionId)
                this.notifyOnce({
                  ...notice,
                  scope: this.scope,
                  sessionId: this.model.sessionId,
                });
            }
            if (!event.error && !this.model.waitingForSnapshot) {
              if (event.type === "snapshot") {
                this.speech.baseline(this.model.messages);
                this.observeMessages(this.model.messages, true);
              } else if (event.type === "message_upsert" && event.message) {
                this.observeMessages([event.message], false);
                this.speech.observe(event.message);
              }
              if (event.type !== "snapshot")
                this.speech.finish(this.model.activities.chat);
            }
            this.reconcilePending();
            attempt = 0;
            this.tick();
            this.emit();
          },
        );
        if (!signal.aborted) throw new UiError("subscriptionEnded");
      } catch (error) {
        if (signal.aborted) return;
        this.ready = false;
        this.speech.cancel();
        this.state.connection = "reconnecting";
        this.connectionIssue = error;
        this.emit();
        try {
          await delay(
            Math.min(30_000, 1_000 * 2 ** Math.min(attempt++, 5)),
            signal,
          );
        } catch {
          return;
        }
      }
    }
  }

  private presentation(): Presentation | undefined {
    if (!this.scope || !this.model.sessionId) return;
    return this.storage.presentation(`${this.scope}:${this.model.sessionId}`);
  }

  private observeMessages(messages: SessionMessage[], snapshot: boolean): void {
    const presentation = this.presentation();
    if (!presentation) return;
    const baseline = snapshot && !presentation.initialized;
    for (const message of messages) {
      if (
        message.source !== "proactive" ||
        message.role !== "assistant" ||
        !message.completed
      )
        continue;
      if (!Object.hasOwn(presentation.receipts, message.id)) {
        presentation.receipts[message.id] = {
          read: baseline,
          presented: baseline,
        };
        this.storage.changed();
      }
    }
    if (!presentation.initialized) {
      presentation.initialized = true;
      this.storage.changed();
    }
  }

  private tick(): void {
    const now = Date.now();
    this.speech.tick(now);
    let settingsChanged = false;
    if (
      this.state.settings.quietUntil &&
      this.state.settings.quietUntil <= now
    ) {
      this.state.settings.quietUntil = 0;
      this.storage.changed();
      settingsChanged = true;
    }
    const receipts = this.presentation()?.receipts ?? {};
    for (const message of this.model.messages) {
      const receipt = receipts[message.id];
      if (!receipt || receipt.read || receipt.suppressed || !message.completed)
        continue;
      if (receipt.snoozeUntil && receipt.snoozeUntil > now) continue;
      if (!receipt.presented || receipt.snoozeUntil)
        this.bubbles.offer({
          id: message.id,
          text: message.content,
          snoozed: Boolean(receipt.snoozeUntil),
        });
    }
    const before = this.state.bubble;
    const native =
      (!this.companionVisible || !this.companionReady) &&
      this.notificationEligible();
    const bubble = this.bubbles.tick(
      now,
      this.state.chatVisible ||
        !(native || (this.companionVisible && this.companionReady)) ||
        this.behavior.dragging ||
        Boolean(this.state.pending?.streaming) ||
        this.behavior.foregroundActive ||
        this.model.status === "active" ||
        this.state.settings.quietUntil > now ||
        !this.ready,
    );
    this.state.bubble =
      this.companionVisible && this.companionReady ? bubble : null;
    this.bubbles.holdForSpeech(
      Boolean(bubble && this.state.speech.playback?.messageId === bubble.id),
      now,
    );
    if (
      this.state.speech.playback?.kind === "reminder" &&
      this.state.speech.playback.messageId !== this.state.bubble?.id
    )
      this.speech.stop();
    if (native && bubble && this.model.sessionId) {
      const sessionId = this.model.sessionId;
      this.notifyOnce(
        {
          id: `reminder:${sessionId}:${bubble.id}:${bubble.presentationId}`,
          scope: this.scope,
          sessionId,
          messageId: bubble.id,
          title: translate(this.state.settings.language, "reminderTitle"),
          body: bubble.text.replace(/\s+/g, " ").slice(0, 140),
        },
        () => {
          if (
            !this.confirmPresentation(
              sessionId,
              bubble.id,
              bubble.presentationId,
            )
          )
            return;
          this.bubbles.dismiss(bubble.id);
          this.tick();
        },
      );
    }
    const previous = this.state.behavior;
    this.resolveBehavior();
    if (
      before !== this.state.bubble ||
      settingsChanged ||
      previous !== this.state.behavior
    )
      this.emit();
  }

  private emit(): void {
    this.state.revision++;
    this.state.sessionId = this.model.sessionId;
    this.state.status = this.model.status;
    this.state.messages = this.model.messages;
    const receipts = this.presentation()?.receipts ?? {};
    this.state.unread = this.model.messages.filter(
      (message) =>
        message.source === "proactive" &&
        message.completed &&
        receipts[message.id]?.read === false,
    ).length;
    this.state.connectionError =
      this.connectionIssue === null
        ? ""
        : errorText(this.connectionIssue, this.state.settings.language);
    if (this.state.pending && this.pendingIssue !== null)
      this.state.pending.error = errorText(
        this.pendingIssue,
        this.state.settings.language,
      );
    this.resolveBehavior();
    this.publish(this.state);
  }

  private resolveBehavior(): void {
    this.state.behavior = this.behavior.resolve(
      this.state,
      this.companionVisible,
    );
    this.state.surface = this.state.bubble ?? this.state.behavior.notice;
  }

  private notificationEligible(): boolean {
    return (
      !this.closed &&
      this.state.settings.notifications &&
      this.state.connection === "connected" &&
      !this.state.chatFocused &&
      this.state.settings.quietUntil <= Date.now()
    );
  }

  private notifyOnce(notice: SystemNotice, shown: () => void = () => {}): void {
    if (!this.notificationEligible() || this.notified.has(notice.id)) return;
    this.notified.add(notice.id);
    this.notify(notice, () => {
      if (this.canOpenNotice(notice)) shown();
    });
  }

  canOpenNotice(notice: SystemNotice): boolean {
    return (
      !this.closed &&
      notice.scope === this.scope &&
      notice.sessionId === this.model.sessionId
    );
  }

  bubbleShown(sessionId: string, id: string, presentationId: number): void {
    if (
      this.closed ||
      !this.companionVisible ||
      !this.companionReady ||
      this.state.bubble?.id !== id
    )
      return;
    if (this.confirmPresentation(sessionId, id, presentationId)) {
      const message = this.model.messages.find((message) => message.id === id);
      if (message) this.speech.speak(message, "reminder");
    }
  }

  private confirmPresentation(
    sessionId: string,
    id: string,
    presentationId: number,
  ): boolean {
    if (
      sessionId !== this.model.sessionId ||
      !this.bubbles.presented(id, presentationId, Date.now())
    )
      return false;
    const receipt = this.presentation()?.receipts[id];
    if (!receipt) return false;
    receipt.presented = true;
    delete receipt.snoozeUntil;
    this.storage.changed();
    return true;
  }

  setChatVisible(visible: boolean, messageId?: string): void {
    this.state.chatVisible = visible;
    if (!visible) this.state.chatFocused = false;
    if (!visible) {
      this.speech.cancelCapture();
      if (this.state.speech.playback?.kind === "manual") this.speech.stop();
    }
    this.state.focusMessageId = messageId ?? null;
    this.tick();
    this.emit();
  }

  setChatFocused(focused: boolean): void {
    this.state.chatFocused = focused;
    if (!focused && this.state.speech.capture?.phase === "recording")
      this.speech.cancelCapture();
    if (focused) this.behavior.touch();
    this.tick();
    this.emit();
  }

  interact(): void {
    this.behavior.touch();
    this.tick();
  }
  drag(phase: "start" | "move" | "end"): void {
    this.behavior.dragging = phase === "move";
    this.behavior.touch();
    this.tick();
  }
  preview(clip: string): void {
    this.behavior.play(clip);
    this.tick();
  }
  reportPlayback(id: number, status: PlaybackStatus): void {
    this.behavior.report(id, status);
    this.tick();
  }

  setCompanionVisible(visible: boolean): void {
    this.companionVisible = visible;
    this.tick();
  }

  setCompanionReady(ready: boolean): void {
    this.companionReady = ready;
    this.tick();
  }

  settings(update: SettingsUpdate): Promise<void> {
    const write = this.settingsWrites.then(() => this.applySettings(update));
    this.settingsWrites = write.catch(() => {});
    return write;
  }

  private async applySettings(update: SettingsUpdate): Promise<void> {
    if (this.closed) throw new UiError("closed");
    if (update.endpoint !== undefined && this.state.pending?.streaming)
      throw new UiError("stopBeforeEndpoint");
    const settings = this.storage.settings;
    const endpoint =
      update.endpoint === undefined
        ? settings.endpoint
        : normalizeEndpoint(update.endpoint);
    if (
      update.scale !== undefined &&
      (!Number.isFinite(update.scale) ||
        update.scale < 0.7 ||
        update.scale > 1.5)
    )
      throw new UiError("invalidScale");
    if (
      update.quietUntil !== undefined &&
      (!Number.isFinite(update.quietUntil) ||
        update.quietUntil < 0 ||
        update.quietUntil > Date.now() + 86_400_000)
    )
      throw new UiError("invalidQuiet");
    const changedEndpoint = endpoint !== settings.endpoint;
    if (update.language !== undefined && !languages.includes(update.language))
      throw new UiError("invalidLanguage");
    if (
      update.notifications !== undefined &&
      typeof update.notifications !== "boolean"
    )
      throw new UiError("invalidNotifications");
    for (const key of [
      "voiceAutoSend",
      "voiceEnabled",
      "voiceReplies",
      "voiceReminders",
    ] as const)
      if (update[key] !== undefined && typeof update[key] !== "boolean")
        throw new UiError("invalidVoiceSettings");
    if (
      update.voiceVolume !== undefined &&
      (!Number.isFinite(update.voiceVolume) ||
        update.voiceVolume < 0 ||
        update.voiceVolume > 1)
    )
      throw new UiError("invalidVoiceSettings");
    this.storage.settings = { ...settings, ...update, endpoint };
    this.storage.changed();
    try {
      await this.storage.flush();
    } catch (error) {
      this.storage.settings = settings;
      this.storage.changed();
      throw error;
    }
    if (this.closed) return;
    this.state.settings = this.storage.settings;
    if (changedEndpoint) this.restart();
    else {
      this.tick();
      this.emit();
    }
  }

  markRead(sessionId: string, ids: string[]): void {
    if (
      sessionId !== this.model.sessionId ||
      !this.state.chatVisible ||
      !this.state.chatFocused
    )
      return;
    const presentation = this.presentation();
    let changed = false;
    for (const id of ids.slice(0, 100)) {
      const receipt = presentation?.receipts[id];
      if (!receipt || receipt.read) continue;
      receipt.read = true;
      changed = true;
      delete receipt.snoozeUntil;
      this.bubbles.dismiss(id);
      this.storage.changed();
    }
    if (!changed) return;
    this.tick();
    this.emit();
  }

  bubbleAction(
    id: string,
    action: "dismiss" | "snooze" | "pause",
    paused = false,
  ): void {
    if (id !== this.bubbles.active?.id) return;
    if (action === "pause") this.bubbles.pause(id, paused, Date.now());
    else {
      if (this.state.speech.playback?.messageId === id) this.speech.stop();
      const receipt: Receipt | undefined = this.presentation()?.receipts[id];
      if (receipt) {
        receipt.presented = true;
        if (action === "snooze") receipt.snoozeUntil = Date.now() + 10 * 60_000;
        this.storage.changed();
      }
      this.bubbles.dismiss(id);
    }
    this.tick();
    this.emit();
  }

  async newSession(): Promise<void> {
    await this.settingsWrites;
    if (
      !this.scope ||
      !this.lifecycle ||
      this.lifecycle.signal.aborted ||
      this.model.status === "active" ||
      this.state.pending?.streaming
    )
      throw new UiError("sessionBusy");
    this.speech.cancel();
    await this.client().call("pa_new_session", {}, this.lifecycle!.signal);
    this.state.pending = null;
    this.emit();
  }

  async send(query: string): Promise<void> {
    await this.settingsWrites;
    query = query.trim();
    if (!query || query.length > 100_000) throw new UiError("invalidQuery");
    if (this.closed || !this.ready || this.model.status === "error")
      throw new UiError("sessionUnavailable");
    if (this.state.pending?.streaming || this.model.status === "active")
      throw new UiError("busy");
    this.speech.cancel();
    const pending = {
      id: randomUUID(),
      query,
      text: "",
      sessionId: this.model.sessionId,
      streaming: true,
    };
    this.state.pending = pending;
    this.pendingIssue = null;
    const generation = this.generation;
    const controller = new AbortController();
    this.stream = controller;
    let progress: ReplyProgress = { text: "", blocks: new Map() };
    this.tick();
    this.emit();
    try {
      await this.client().stream(
        "pa_chat",
        { query },
        AbortSignal.any([controller.signal, this.lifecycle!.signal]),
        (chunk) => {
          if (generation !== this.generation) return;
          progress = reduceReply(progress, chunk);
          if (this.state.pending !== pending) return;
          Object.assign(pending, {
            text: progress.text,
            turnId: progress.turnId,
            sessionId: progress.sessionId ?? pending.sessionId,
          });
          this.emit();
        },
      );
      if (progress.error)
        throw typeof progress.error === "string"
          ? new Error(progress.error)
          : progress.error;
      if (progress.status !== "completed") throw new UiError("replyIncomplete");
    } catch (error) {
      if (generation === this.generation && this.state.pending === pending) {
        this.pendingIssue = controller.signal.aborted
          ? new UiError("interrupted")
          : error;
      }
    } finally {
      if (generation === this.generation && this.state.pending === pending) {
        pending.streaming = false;
        this.stream = null;
        this.emit();
        const cursor = this.model.cursor;
        try {
          const snapshot = await this.client().call<SessionSnapshot>(
            "session_read",
            { session_id: pending.sessionId },
            this.lifecycle!.signal,
          );
          if (
            generation === this.generation &&
            snapshot.session_id === this.model.sessionId &&
            cursor === this.model.cursor
          )
            this.model.snapshot(snapshot);
        } catch {
          /* The subscription restores accepted messages after a disconnect. */
        }
        if (generation !== this.generation) return;
        this.reconcilePending();
        this.tick();
        this.emit();
      }
    }
  }

  private reconcilePending(): void {
    const pending = this.state.pending;
    if (!pending || pending.streaming) return;
    const native = this.model.messages.find(
      (message) =>
        message.role === "assistant" &&
        pending.turnId &&
        message.turn_id === pending.turnId &&
        message.completed,
    );
    if (native && !pending.error) this.state.pending = null;
  }

  stopReply(): void {
    this.speech.cancel();
    this.stream?.abort();
  }

  async close(): Promise<void> {
    this.closed = true;
    clearInterval(this.clock);
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    await this.speech.close();
    await this.settingsWrites;
    await this.running;
    await this.storage.flush();
  }
}
