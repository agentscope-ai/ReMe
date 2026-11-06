import { createHash, randomUUID } from "node:crypto";
import { BubbleQueue } from "./bubbles.ts";
import { BehaviorController } from "./behavior.ts";
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
} from "../shared/protocol.ts";

const detail = (error: unknown) =>
  error instanceof Error ? error.message : String(error);

export class DesktopController {
  readonly model = new SessionModel();
  readonly bubbles: BubbleQueue;
  readonly state: DesktopState;
  private readonly behavior: BehaviorController;
  private scope = "";
  private lifecycle: AbortController | null = null;
  private stream: AbortController | null = null;
  private running: Promise<void> = Promise.resolve();
  private clock: ReturnType<typeof setInterval> | undefined;
  private ready = false;
  private companionVisible = true;
  private generation = 0;
  private endpoint: string;
  private settingsWrites: Promise<void> = Promise.resolve();
  private closed = false;
  private flushedAt = 0;

  constructor(
    readonly storage: Storage,
    private readonly publish: (state: DesktopState) => void,
    manifest: AnimationManifest,
  ) {
    this.behavior = new BehaviorController(manifest);
    this.endpoint = storage.settings.endpoint;
    this.state = {
      connection: "connecting",
      connectionError: "",
      settings: storage.settings,
      workspace: "",
      sessionId: null,
      status: null,
      messages: [],
      pending: null,
      bubble: null,
      unread: 0,
      chatVisible: false,
      chatFocused: false,
      focusMessageId: null,
      behavior: this.behavior.value,
    };
    this.bubbles = new BubbleQueue(
      (id) => {
        const receipt = this.presentation()?.receipts[id];
        if (receipt) {
          receipt.presented = true;
          delete receipt.snoozeUntil;
          this.storage.changed();
        }
      },
      10 * 60_000,
      (id) => {
        const receipt = this.presentation()?.receipts[id];
        if (receipt) {
          receipt.suppressed = true;
          this.storage.changed();
        }
      },
    );
  }

  start(): void {
    this.restart();
    this.clock = setInterval(() => {
      this.tick();
      if (Date.now() - this.flushedAt < 1000) return;
      this.flushedAt = Date.now();
      void this.storage.flush().catch((error) => {
        this.state.connectionError = `保存桌面设置失败：${detail(error)}`;
        this.emit();
      });
    }, 100);
  }

  private restart(): void {
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    this.endpoint = this.storage.settings.endpoint;
    this.ready = false;
    this.model.clear();
    this.behavior.reset();
    this.bubbles.clear();
    this.state.pending = null;
    this.state.workspace = "";
    this.scope = "";
    this.state.connection = "connecting";
    this.state.connectionError = "";
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
            throw new Error(
              `服务未启用 ${job}，请使用 reme start config=proactive 启动`,
            );
          }
        }
        if (typeof config.workspace_dir !== "string")
          throw new Error("ReMe 未返回工作区路径");
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
            if (chunk.chunk_type === "error")
              throw new Error(
                typeof chunk.chunk === "string" ? chunk.chunk : "订阅出错",
              );
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
              this.behavior.reset();
              this.bubbles.clear();
              this.state.focusMessageId = null;
            }
            if (event.type === "snapshot" || event.type === "status_changed")
              this.behavior.observe(
                this.model.activities,
                event.type !== "snapshot",
              );
            this.ready = !this.model.waitingForSnapshot;
            this.state.connection = this.ready ? "connected" : "reconnecting";
            this.state.connectionError = event.error ?? "";
            if (!event.error && !this.model.waitingForSnapshot) {
              if (event.type === "snapshot")
                this.observeMessages(this.model.messages, true);
              else if (event.type === "message_upsert" && event.message)
                this.observeMessages([event.message], false);
            }
            this.reconcilePending();
            attempt = 0;
            this.tick();
            this.emit();
          },
        );
        if (!signal.aborted)
          throw new Error("主动消息订阅已结束，正在重新连接");
      } catch (error) {
        if (signal.aborted) return;
        this.ready = false;
        this.state.connection = "reconnecting";
        this.state.connectionError = detail(error);
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
    let settingsChanged = false;
    if (
      this.storage.settings.quietUntil &&
      this.storage.settings.quietUntil <= now
    ) {
      this.storage.settings.quietUntil = 0;
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
    const bubble = this.bubbles.tick(
      now,
      this.state.chatVisible ||
        !this.companionVisible ||
        Boolean(this.state.pending?.streaming) ||
        this.behavior.foregroundActive ||
        this.model.status === "active" ||
        this.storage.settings.quietUntil > now ||
        !this.ready,
    );
    this.state.bubble = bubble;
    const previous = this.state.behavior;
    this.state.behavior = this.behavior.resolve(
      this.state,
      this.companionVisible,
    );
    if (
      before !== bubble ||
      settingsChanged ||
      previous !== this.state.behavior
    )
      this.emit();
  }

  private emit(): void {
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
    this.state.settings = { ...this.storage.settings };
    this.state.behavior = this.behavior.resolve(
      this.state,
      this.companionVisible,
    );
    this.publish(this.state);
  }

  setChatVisible(visible: boolean, messageId?: string): void {
    this.state.chatVisible = visible;
    if (!visible) this.state.chatFocused = false;
    this.state.focusMessageId = messageId ?? null;
    this.tick();
    this.emit();
  }

  setChatFocused(focused: boolean): void {
    this.state.chatFocused = focused;
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
  animationEnded(id: number): void {
    this.behavior.ended(id);
    this.tick();
  }

  setCompanionVisible(visible: boolean): void {
    this.companionVisible = visible;
    this.tick();
  }

  settings(update: {
    endpoint?: string;
    scale?: number;
    quietUntil?: number;
  }): Promise<void> {
    const write = this.settingsWrites.then(() => this.applySettings(update));
    this.settingsWrites = write.catch(() => {});
    return write;
  }

  private async applySettings(update: {
    endpoint?: string;
    scale?: number;
    quietUntil?: number;
  }): Promise<void> {
    if (this.closed) throw new Error("桌面客户端已关闭");
    if (update.endpoint !== undefined && this.state.pending?.streaming)
      throw new Error("请先停止当前回复，再切换服务");
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
      throw new Error("角色大小需要在 70%–150% 之间");
    if (
      update.quietUntil !== undefined &&
      (!Number.isFinite(update.quietUntil) ||
        update.quietUntil < 0 ||
        update.quietUntil > Date.now() + 86_400_000)
    )
      throw new Error("免打扰时间无效");
    const changedEndpoint = endpoint !== settings.endpoint;
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
    if (changedEndpoint) this.restart();
    else {
      this.tick();
      this.emit();
    }
  }

  markRead(sessionId: string, ids: string[]): void {
    if (sessionId !== this.model.sessionId || !this.state.chatVisible) return;
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
      throw new Error("请等待连接建立、当前会话空闲后再新建");
    await this.client().call("pa_new_session", {}, this.lifecycle!.signal);
    this.state.pending = null;
    this.emit();
  }

  async send(query: string): Promise<void> {
    await this.settingsWrites;
    query = query.trim();
    if (!query || query.length > 100_000)
      throw new Error("消息不能为空，且不超过 100000 字符");
    if (this.closed || !this.ready || this.model.status === "error")
      throw new Error("会话暂不可用，请检查连接或新建对话");
    if (this.state.pending?.streaming || this.model.status === "active")
      throw new Error("Remy 正在忙，请稍等或先停止回复");
    const pending = {
      id: randomUUID(),
      query,
      text: "",
      sessionId: this.model.sessionId,
      streaming: true,
    };
    this.state.pending = pending;
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
      if (progress.error || progress.status !== "completed")
        throw new Error(
          progress.error ?? "回复未完成，可查看已同步的会话后再继续",
        );
    } catch (error) {
      if (generation === this.generation && this.state.pending === pending) {
        Object.assign(pending, {
          error: controller.signal.aborted ? "回复已停止" : detail(error),
        });
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
    this.stream?.abort();
  }

  async close(): Promise<void> {
    this.closed = true;
    clearInterval(this.clock);
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    await this.settingsWrites;
    await this.running;
    await this.storage.flush();
  }
}
