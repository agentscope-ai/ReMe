import { errorText, languages, translate, UiError } from "./i18n.ts";
import { BubbleQueue } from "./bubbles.ts";
import type { Reminder } from "./bubbles.ts";
import { PresentationController } from "./presentation.ts";
import { displayMessages } from "./messages.ts";
import { surfaceKey, replyText } from "./surface.ts";
import { NotificationPolicy } from "./notifications.ts";
import { emptySpeech, SpeechController } from "./speech-controller.ts";
import type { SystemNotice } from "./notifications.ts";
import { delay, ReMeClient } from "./reme-client.ts";
import { SessionModel } from "./session-model.ts";
import { normalizeEndpoint, type ClientStorage } from "./settings.ts";
import { requestWorkspace } from "./workspace.ts";
import { emptyReply, reduceReply } from "./transcript.ts";
import type { ReplyProgress } from "./transcript.ts";
import type {
  StudioState,
  PendingTurn,
  ClientInfo,
  Presentation,
  Receipt,
  AgentMessage,
  Activities,
  Behavior,
  PlaybackStatus,
  SettingsUpdate,
  StudioCommand,
  StudioPage,
  WorkspaceRequest,
  ItemTarget,
  TaskReminder,
} from "./protocol.ts";
import { activityLevels, sleepDurations } from "./protocol.ts";

export function validateItemTarget(
  value: unknown,
): asserts value is ItemTarget {
  const target = value as ItemTarget | null;
  if (
    !target ||
    !["message", "mail"].includes(target.kind) ||
    typeof target.id !== "string" ||
    !target.id ||
    target.id.length > 8192 ||
    target.id.includes("\0") ||
    (target.kind === "mail" &&
      (typeof target.agentName !== "string" ||
        !target.agentName ||
        target.agentName.length > 128))
  )
    throw new UiError("invalidAction");
}

export interface CompanionBehavior {
  value: Behavior;
  dragging: boolean;
  readonly foregroundActive: boolean;
  reset(): void;
  observe(activities: Activities, live: boolean, now: number): void;
  touch(): void;
  play(clip: string): void;
  report(id: number, status: PlaybackStatus): void;
  resolve(
    state: StudioState & { reminderId: string | null },
    visible: boolean,
  ): Behavior;
}

export class ClientController {
  readonly model = new SessionModel();
  readonly bubbles: BubbleQueue;
  // Paint acknowledgements must stay distinct when the same item moves between queues.
  private bubbleSequence = 0;
  private readonly nextPresentationId = () => ++this.bubbleSequence;
  private readonly instantBubbles = new BubbleQueue(
    0,
    () => {},
    Infinity,
    false,
    this.nextPresentationId,
  );
  private readonly replayBubbles = new BubbleQueue(
    0,
    () => {},
    Infinity,
    true,
    this.nextPresentationId,
  );
  private replay: { target: ItemTarget; message: AgentMessage } | null = null;
  private replayRequest = 0;
  private menuOpen = false;
  private menuSequence = 0;
  readonly state: StudioState;
  readonly speech: SpeechController;
  private readonly presentationController = new PresentationController();
  private messageInputs: unknown[] = [];
  private visibleReminder: Reminder | null = null;
  private scope = "";
  private lifecycle: AbortController | null = null;
  private stream: AbortController | null = null;
  private running: Promise<void> = Promise.resolve();
  private clock: ReturnType<typeof setInterval> | undefined;
  private ready = false;
  private companionVisible: boolean;
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
  private chatPending: PendingTurn | null = null;
  private selectedTask?: string;
  private childWatch: AbortController | null = null;
  private childWatching: Promise<void> = Promise.resolve();
  private children = new Map<
    string,
    {
      model: SessionModel;
      pending: PendingTurn | null;
      reply: ReplyProgress;
      error: unknown;
      connected: boolean;
      stream?: AbortController;
      sending?: Promise<void>;
    }
  >();

  constructor(
    readonly storage: ClientStorage,
    private readonly publish: (state: StudioState) => void,
    private readonly behavior?: CompanionBehavior,
    private readonly notify: (
      notice: SystemNotice,
      shown: () => void,
    ) => void = () => {},
    command: (command: StudioCommand) => void = () => {},
    private readonly fetchRequest: typeof fetch = (...args) =>
      globalThis.fetch(...args),
  ) {
    this.companionVisible = Boolean(behavior);
    this.endpoint = storage.settings.endpoint;
    this.state = {
      revision: 0,
      connection: "connecting",
      connectionError: "",
      settings: storage.settings,
      clientInfo: null,
      selectedAgentName: "Remy",
      agents: [],
      sessionId: null,
      status: null,
      messages: [],
      reminders: [],
      pending: null,
      surface: null,
      unread: 0,
      readMessageIds: [],
      activePage: "remy",
      documentTarget: null,
      chatVisible: false,
      chatFocused: false,
      focusTarget: null,
      focusRequest: 0,
      behavior: behavior?.value ?? {
        animation: { id: 0, clip: "idle", mode: "hold" },
        label: "",
        error: "",
      },
      speech: emptySpeech(),
    };
    this.bubbles = new BubbleQueue(
      10 * 60_000,
      (id) => {
        const receipt = this.presentation()?.receipts[id];
        if (receipt) {
          receipt.suppressed = true;
          this.storage.changed();
        }
      },
      3,
      true,
      this.nextPresentationId,
    );
    this.speech = new SpeechController(
      () => ({
        sessionId: this.state.sessionId,
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
    this.childWatch?.abort();
    for (const child of this.children.values()) child.stream?.abort();
    this.children.clear();
    this.selectedTask = undefined;
    this.state.selectedAgentName = "Remy";
    this.state.agents = [];
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    this.endpoint = this.storage.settings.endpoint;
    this.ready = false;
    this.model.clear();
    this.behavior?.reset();
    this.presentationController.reset();
    this.notificationPolicy.reset();
    this.notified.clear();
    this.bubbles.clear();
    this.instantBubbles.clear();
    this.replayBubbles.clear();
    this.replay = null;
    this.replayRequest++;
    this.menuOpen = false;
    this.visibleReminder = null;
    this.state.surface = null;
    this.chatPending = null;
    this.pendingIssue = null;
    this.state.clientInfo = null;
    this.state.documentTarget = null;
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
    return new ReMeClient(this.endpoint, this.fetchRequest);
  }

  private async connect(
    generation: number,
    signal: AbortSignal,
  ): Promise<void> {
    let attempt = 0;
    while (!signal.aborted) {
      try {
        const client = this.client();
        const config = await client.call<ClientInfo>(
          "get_client_info",
          {},
          signal,
        );
        if (signal.aborted) return;
        for (const job of [
          "watch_agent_session",
          "ask_agent",
          "create_chat_session",
          "list_task_reminders",
          "read_agent_reminder",
          "list_agent_inbox",
        ]) {
          if (!config.jobs.includes(job))
            throw new UiError("missingJob", { job });
        }
        this.speech.configure(
          config.asr,
          config.tts,
          JSON.stringify([
            client.endpoint,
            config.workspace_dir,
            config.speech_revision,
          ]),
        );
        const scope = this.storage.scope(client.endpoint, config.workspace_dir);
        if (scope !== this.scope) {
          this.model.clear();
          this.bubbles.clear();
          this.instantBubbles.clear();
          this.scope = scope;
        }
        this.state.clientInfo = config;
        await client.stream(
          "watch_agent_session",
          { agent_name: "Remy" },
          signal,
          (chunk) => {
            if (signal.aborted || generation !== this.generation) return;
            if (chunk.chunk_type === "error") {
              if (typeof chunk.chunk === "string") throw new Error(chunk.chunk);
              throw new UiError("subscriptionError");
            }
            if (chunk.chunk_type !== "data" || !chunk.chunk) return;
            const previous = this.model.sessionId;
            const event = this.model.apply(chunk.chunk);
            if (event.type !== "heartbeat" && event.agents) {
              this.state.agents = event.agents;
              this.syncAgentSelection();
            }
            const live = this.ready && previous === this.model.sessionId;
            if (
              event.type === "heartbeat" &&
              this.state.connection === "connected"
            )
              return;
            if (previous !== this.model.sessionId) {
              this.speech.reset();
              this.behavior?.reset();
              this.presentationController.reset();
              this.notificationPolicy.reset();
              this.notified.clear();
              this.bubbles.clear();
              this.instantBubbles.clear();
              this.replayBubbles.clear();
              this.replay = null;
              this.replayRequest++;
              this.menuOpen = false;
              this.visibleReminder = null;
              this.state.focusTarget = null;
            }
            if (event.type !== "heartbeat") {
              this.presentationController.observe(
                this.model.activities.Remy,
                live,
              );
              this.behavior?.observe(this.model.activities, live, Date.now());
            }
            this.ready = true;
            this.state.connection = "connected";
            this.connectionIssue = null;
            if (event.type !== "heartbeat") {
              const notice = this.notificationPolicy.observe(
                this.model.activities.Remy,
                live,
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
            if (event.type !== "heartbeat") {
              if (event.type === "snapshot") {
                if (live) {
                  for (const message of this.model.messages)
                    this.speech.observe(message);
                } else this.speech.baseline(this.model.messages);
                this.observeMessages(this.model.messages, !live);
              } else {
                this.observeMessages(event.messages, false);
                for (const message of event.messages)
                  this.speech.observe(message);
              }
              if (live) this.speech.finish(this.model.activities.Remy);
            }
            this.syncPending(this.model, this.chatPending);
            this.reconcilePending();
            attempt = 0;
            this.tick();
            this.emit();
          },
        );
        if (!signal.aborted) throw new UiError("subscriptionEnded");
      } catch (error) {
        if (signal.aborted) return;
        console.warn("Remy subscription failed:", error);
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

  private notificationMessage(
    id: string | undefined,
  ): AgentMessage | undefined {
    return this.model.messages.find((message) => message.id === id);
  }

  private notificationMessages(): AgentMessage[] {
    return this.model.messages.filter(
      (message) =>
        message.source === "proactive" &&
        message.role === "assistant" &&
        message.completed &&
        message.content.trim().length > 0,
    );
  }

  private notificationPresentation(
    message: AgentMessage,
  ): Presentation | undefined {
    const session = message.origin_session_id ?? this.model.sessionId;
    return this.scope && session
      ? this.storage.presentation(`${this.scope}:${session}`)
      : undefined;
  }

  private observeMessages(messages: AgentMessage[], snapshot: boolean): void {
    const current = this.presentation();
    const initialize = new Set<Presentation>();
    for (const message of messages) {
      if (
        message.source !== "proactive" ||
        message.role !== "assistant" ||
        !message.completed
      )
        continue;
      const presentation = this.notificationPresentation(message);
      if (!presentation) continue;
      const baseline =
        snapshot &&
        !presentation.initialized &&
        message.delivery !== "immediate";
      if (snapshot) initialize.add(presentation);
      if (!Object.hasOwn(presentation.receipts, message.id)) {
        presentation.receipts[message.id] = {
          presented: baseline,
          userRead: false,
        };
        this.storage.changed();
      }
    }
    if (snapshot && current) initialize.add(current);
    for (const presentation of initialize) {
      if (!presentation.initialized) {
        presentation.initialized = true;
        this.storage.changed();
      }
    }
  }

  private tick(): void {
    const now = Date.now();
    this.synchronize();
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
    this.instantBubbles.retain(
      new Set(
        this.model.messages
          .filter(
            (message) =>
              message.delivery === "immediate" && message.notification_eligible,
          )
          .map((message) => message.id),
      ),
    );
    if (this.bubbles.active) {
      const active = this.model.messages.find(
        (message) => message.id === this.bubbles.active?.id,
      );
      if (!active || !active.notification_eligible)
        this.bubbles.dismiss(this.bubbles.active.id);
    }
    for (const message of this.notificationMessages()) {
      if (
        message.delivery !== "immediate" &&
        (message.origin_session_id ?? this.model.sessionId) !==
          this.model.sessionId
      )
        continue;
      const receipt =
        this.notificationPresentation(message)?.receipts[message.id];
      if (
        !receipt ||
        receipt.suppressed ||
        (receipt.userRead && !receipt.snoozeUntil)
      )
        continue;
      if (!message.notification_eligible) {
        receipt.suppressed = true;
        this.bubbles.dismiss(message.id);
        this.storage.changed();
        continue;
      }
      if (receipt.snoozeUntil && receipt.snoozeUntil > now) continue;
      if (!receipt.presented || receipt.snoozeUntil)
        (message.delivery === "immediate"
          ? this.instantBubbles
          : this.bubbles
        ).offer({
          id: message.id,
          text: message.content,
          snoozed: Boolean(receipt.snoozeUntil),
        });
    }
    const before = surfaceKey(this.state.surface);
    const native =
      (!this.companionVisible || !this.companionReady) &&
      this.notificationEligible();
    const immediateNative =
      (!this.companionVisible || !this.companionReady) &&
      this.notificationEligible(true);
    const replay = this.replayBubbles.tick(
      now,
      this.menuOpen ||
        !this.companionVisible ||
        !this.companionReady ||
        Boolean(this.behavior?.dragging),
    );
    const immediate = this.instantBubbles.tick(
      now,
      this.menuOpen ||
        Boolean(replay) ||
        !(immediateNative || (this.companionVisible && this.companionReady)) ||
        Boolean(this.behavior?.dragging) ||
        !this.ready,
    );
    const ordinary = this.bubbles.tick(
      now,
      this.menuOpen ||
        Boolean(replay) ||
        !(native || (this.companionVisible && this.companionReady)) ||
        Boolean(this.behavior?.dragging) ||
        Boolean(this.chatPending?.streaming) ||
        this.behavior?.foregroundActive ||
        (this.companionVisible &&
          this.companionReady &&
          this.presentationController.foregroundActive) ||
        this.model.status === "active" ||
        this.state.settings.quietUntil > now ||
        !this.ready ||
        Boolean(immediate),
    );
    const bubble = replay ?? immediate ?? ordinary;
    this.visibleReminder =
      this.companionVisible && this.companionReady ? bubble : null;
    this.resolvePresentation();
    this.bubbles.holdForSpeech(
      Boolean(
        bubble &&
        this.state.speech.playback?.messageId === bubble.id &&
        this.state.speech.playback.phase !== "error",
      ),
      now,
    );
    this.instantBubbles.holdForSpeech(
      Boolean(
        immediate &&
        this.state.speech.playback?.messageId === immediate.id &&
        this.state.speech.playback.phase !== "error",
      ),
      now,
    );
    this.replayBubbles.holdForSpeech(
      Boolean(
        replay &&
        this.state.speech.playback?.messageId === replay.id &&
        this.state.speech.playback.phase !== "error",
      ),
      now,
    );
    if (
      !this.menuOpen &&
      this.state.speech.playback?.kind === "reminder" &&
      this.state.speech.playback.phase !== "error" &&
      (this.state.surface?.kind !== "message" ||
        this.state.speech.playback.messageId !== this.state.surface.message.id)
    )
      this.speech.stop();
    if (
      !replay &&
      (immediate ? immediateNative : native) &&
      bubble &&
      this.model.sessionId
    ) {
      const sessionId = this.model.sessionId;
      this.notifyOnce(
        {
          id: `reminder:${sessionId}:${bubble.id}:${bubble.presentationId}`,
          scope: this.scope,
          sessionId,
          messageId: bubble.id,
          expiresAt: now + 24 * 60 * 60_000,
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
          (immediate ? this.instantBubbles : this.bubbles).dismiss(bubble.id);
          this.tick();
        },
      );
    }
    const previous = this.state.behavior;
    this.resolveBehavior();
    if (
      before !== surfaceKey(this.state.surface) ||
      settingsChanged ||
      previous !== this.state.behavior
    )
      this.emit();
  }

  private emit(): void {
    this.state.revision++;
    this.synchronize();
    this.resolvePresentation();
    this.resolveBehavior();
    this.publish(this.state);
  }

  private synchronize(): void {
    this.state.pending = this.chatPending;
    this.state.sessionId = this.model.sessionId;
    this.state.status = this.model.status;
    const notifications = this.notificationMessages();
    this.state.reminders = displayMessages(
      {
        sessionId: this.model.sessionId,
        messages: notifications,
        pending: null,
      },
      undefined,
      this.state.settings.language,
    );
    this.state.readMessageIds = notifications
      .filter(
        (message) =>
          this.notificationPresentation(message)?.receipts[message.id]
            ?.userRead,
      )
      .map((message) => message.id);
    this.state.unread = this.state.reminders.filter(
      (message) => !this.state.readMessageIds.includes(message.id),
    ).length;
    this.state.connectionError =
      this.connectionIssue === null
        ? ""
        : errorText(this.connectionIssue, this.state.settings.language);
    if (this.chatPending && this.pendingIssue !== null)
      this.chatPending.error = errorText(
        this.pendingIssue,
        this.state.settings.language,
      );
    const pending = this.chatPending;
    const inputs = [
      this.model.messages,
      this.model.sessionId,
      this.model.activities.Remy,
      this.state.settings.language,
      pending?.id,
      pending?.turnId,
      pending?.sessionId,
      pending?.outcome,
      pending?.streaming,
      pending?.error,
    ];
    if (inputs.some((value, index) => value !== this.messageInputs[index])) {
      this.messageInputs = inputs;
      this.state.messages = displayMessages(
        {
          sessionId: this.model.sessionId,
          messages: this.model.messages,
          pending,
        },
        this.model.activities.Remy,
        this.state.settings.language,
      );
    }
    const selected = this.state.selectedAgentName ?? "Remy";
    const child = this.children.get(selected);
    if (selected !== "Remy" && child) {
      this.state.sessionId = child.model.sessionId;
      this.state.status = child.connected ? child.model.status : "error";
      this.state.pending = child.pending;
      this.state.messages = displayMessages(
        {
          sessionId: child.pending?.sessionId ?? child.model.sessionId,
          messages: child.pending ? [] : child.model.messages,
          pending: child.pending,
          reply: child.reply,
          task: true,
        },
        child.pending ? undefined : child.model.activities[selected],
        this.state.settings.language,
      );
      if (child.error)
        this.state.connectionError = errorText(
          child.error,
          this.state.settings.language,
        );
    }
  }

  private resolvePresentation(): void {
    if (this.menuOpen && this.companionVisible) {
      this.state.surface = {
        kind: "menu",
        id: `companion-menu:${this.menuSequence}`,
        text: "Remy",
      };
      return;
    }
    if (
      this.visibleReminder &&
      this.replay &&
      this.visibleReminder === this.replayBubbles.active
    ) {
      this.state.surface = {
        kind: "message",
        message: displayMessages(
          {
            sessionId: this.model.sessionId,
            messages: [this.replay.message],
            pending: null,
          },
          undefined,
          this.state.settings.language,
        )[0],
        target: this.replay.target,
        canSnooze: false,
        presentationId: this.visibleReminder.presentationId,
        paused: this.visibleReminder.paused,
      };
      return;
    }
    const instant =
      this.visibleReminder && this.notificationMessage(this.visibleReminder.id);
    if (instant?.delivery === "immediate" && this.visibleReminder) {
      this.state.surface = {
        kind: "message",
        target: { kind: "message", id: instant.id },
        canSnooze: true,
        message: {
          id: instant.id,
          createdAt: instant.created_at,
          role: "assistant",
          source: "proactive",
          content: instant.content,
          completed: true,
          streaming: false,
          status: null,
        },
        presentationId: this.visibleReminder.presentationId,
        paused: this.visibleReminder.paused,
      };
      return;
    }
    this.state.surface = this.presentationController.resolve(
      this.state,
      this.model.activities,
      this.visibleReminder,
      this.companionVisible && this.companionReady,
      Boolean(this.behavior?.dragging),
    );
  }

  private resolveBehavior(): void {
    if (!this.behavior) return;
    this.state.behavior = this.behavior.resolve(
      {
        ...this.state,
        reminderId:
          this.state.surface?.kind === "message" &&
          this.state.surface.message.source === "proactive"
            ? this.state.surface.message.id
            : null,
      },
      this.companionVisible && this.companionReady,
    );
  }

  private notificationEligible(immediate = false): boolean {
    return (
      !this.closed &&
      this.state.settings.notifications &&
      this.state.connection === "connected" &&
      !this.state.chatFocused &&
      (immediate || this.state.settings.quietUntil <= Date.now())
    );
  }

  private notifyOnce(notice: SystemNotice, shown: () => void = () => {}): void {
    if (!this.canNotify(notice) || this.notified.has(notice.id)) return;
    this.notified.add(notice.id);
    this.notify(notice, () => {
      if (this.canNotify(notice)) shown();
    });
  }

  canNotify(notice: SystemNotice): boolean {
    const immediate =
      this.notificationMessage(notice.messageId)?.delivery === "immediate";
    if (!this.notificationEligible(immediate) || !this.canOpenNotice(notice))
      return false;
    if (!notice.messageId) return true;
    const message = this.notificationMessage(notice.messageId);
    return Boolean(message && message.notification_eligible);
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
      this.state.surface?.kind !== "message" ||
      this.state.surface.message.source !== "proactive" ||
      this.state.surface.message.id !== id
    )
      return;
    if (
      this.replay &&
      sessionId === this.model.sessionId &&
      this.replayBubbles.presented(id, presentationId, Date.now())
    ) {
      const original =
        this.replay.target.kind === "message"
          ? this.notificationMessage(id)
          : undefined;
      const receipt =
        original && this.notificationPresentation(original)?.receipts[id];
      if (receipt) {
        receipt.presented = true;
        delete receipt.snoozeUntil;
        this.bubbles.dismiss(id);
        this.instantBubbles.dismiss(id);
        this.storage.changed();
      }
      this.speech.speak(this.replay.message, "reminder");
      return;
    }
    if (this.confirmPresentation(sessionId, id, presentationId)) {
      const message = this.notificationMessage(id);
      if (message) {
        const settings = this.state.settings;
        if (
          message.delivery === "immediate" &&
          settings.voiceEnabled &&
          settings.voiceReminders &&
          settings.voiceVolume > 0 &&
          !this.state.speech.capture
        )
          this.speech.stop();
        this.speech.speak(message, "reminder");
      }
    }
  }

  private confirmPresentation(
    sessionId: string,
    id: string,
    presentationId: number,
  ): boolean {
    const message = this.notificationMessage(id);
    if (
      sessionId !== this.model.sessionId ||
      !message ||
      !message.notification_eligible ||
      !(
        this.instantBubbles.active?.id === id
          ? this.instantBubbles
          : this.bubbles
      ).presented(id, presentationId, Date.now())
    )
      return false;
    const receipt =
      message && this.notificationPresentation(message)?.receipts[id];
    if (!receipt) return false;
    receipt.presented = true;
    delete receipt.snoozeUntil;
    this.storage.changed();
    return true;
  }

  setChatVisible(visible: boolean, target?: ItemTarget): void {
    if (target) validateItemTarget(target);
    this.state.chatVisible = visible;
    if (!visible) this.state.chatFocused = false;
    if (!visible) {
      this.speech.cancelCapture();
      if (this.state.speech.playback?.kind === "manual") this.speech.stop();
    }
    if (target) {
      const agentName = target.kind === "mail" ? target.agentName : "Remy";
      this.openAgent(agentName);
    }
    this.state.focusTarget = target ?? null;
    if (target) {
      this.state.focusRequest++;
    }
    this.tick();
    this.emit();
  }

  setChatFocused(focused: boolean): void {
    this.state.chatFocused = focused;
    if (!focused && this.state.speech.capture?.phase !== "transcribing")
      this.speech.cancelCapture();
    if (focused) this.behavior?.touch();
    this.tick();
    this.emit();
  }

  setMenuOpen(open: boolean): void {
    if (open) this.menuSequence++;
    this.menuOpen = open && this.companionVisible;
    this.tick();
    this.emit();
  }

  interact(): void {
    this.behavior?.touch();
    this.tick();
  }
  drag(phase: "start" | "move" | "end"): void {
    if (this.behavior) this.behavior.dragging = phase === "move";
    this.behavior?.touch();
    this.tick();
  }
  preview(clip: string): void {
    this.behavior?.play(clip);
    this.tick();
  }
  reportPlayback(id: number, status: PlaybackStatus): void {
    this.behavior?.report(id, status);
    this.tick();
  }

  setCompanionVisible(visible: boolean): void {
    if (!visible) this.menuOpen = false;
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
    if (
      update.endpoint !== undefined &&
      (this.chatPending?.streaming ||
        [...this.children.values()].some((child) => child.pending?.streaming))
    )
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
    if (
      (update.activityLevel !== undefined &&
        !activityLevels.includes(update.activityLevel)) ||
      (update.sleepMinutes !== undefined &&
        !sleepDurations.includes(update.sleepMinutes))
    )
      throw new UiError("invalidAnimationSettings");
    if (
      update.theme !== undefined &&
      !["system", "light", "dark"].includes(update.theme)
    )
      throw new UiError("invalidTheme");
    if (
      update.transparentBackground !== undefined &&
      typeof update.transparentBackground !== "boolean"
    )
      throw new UiError("invalidTransparency");
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
      !this.state.chatVisible ||
      !this.state.chatFocused ||
      this.state.activePage !== "remy"
    )
      return;
    const messages: AgentMessage[] = [];
    for (const id of ids.slice(0, 100)) {
      const message = this.notificationMessage(id);
      if (
        message &&
        (message.origin_session_id ?? this.model.sessionId) === sessionId
      )
        messages.push(message);
    }
    this.readMessages(messages);
  }

  private readMessages(messages: AgentMessage[]): void {
    let changed = false;
    for (const message of messages) {
      const id = message.id;
      const receipt = this.notificationPresentation(message)?.receipts[id];
      if (!receipt || receipt.userRead) continue;
      receipt.userRead = true;
      changed = true;
      delete receipt.snoozeUntil;
      if (this.bubbles.active?.id !== id) this.bubbles.dismiss(id);
      if (this.instantBubbles.active?.id !== id)
        this.instantBubbles.dismiss(id);
    }
    if (!changed) return;
    this.storage.changed();
    this.tick();
    this.emit();
  }

  selectPage(page: StudioPage): void {
    this.state.activePage = page;
    this.syncAgentSelection();
    if (page !== "remy") this.speech.cancelCapture();
    this.tick();
    this.emit();
  }

  async workspace(request: WorkspaceRequest) {
    if (!this.lifecycle || this.closed) throw new UiError("closed");
    return requestWorkspace(this.client(), request, this.lifecycle.signal);
  }

  markAllRead(): void {
    this.readMessages(this.notificationMessages());
  }

  openDocument(path: string): void {
    this.state.documentTarget = {
      path,
      request: (this.state.documentTarget?.request ?? 0) + 1,
    };
    this.selectPage("knowledge");
  }

  async item(target: ItemTarget): Promise<AgentMessage> {
    validateItemTarget(target);
    if (this.closed) throw new UiError("itemUnavailable");
    if (target.kind === "message") {
      const message = this.notificationMessage(target.id);
      if (
        !message ||
        message.role !== "assistant" ||
        !message.completed ||
        !message.content
      )
        throw new UiError("itemUnavailable");
      return message;
    }
    const generation = this.generation;
    const response = await this.workspace({
      type: "reminder",
      agentName: target.agentName,
      id: target.id,
    });
    const mail = response.answer as TaskReminder;
    if (
      this.closed ||
      generation !== this.generation ||
      !mail?.message ||
      mail.id !== target.id
    )
      throw new UiError("itemUnavailable");
    return {
      id: mail.id,
      role: "assistant",
      source: "proactive",
      content: mail.message,
      completed: true,
      created_at: mail.delivered_at,
      session_id: this.model.sessionId ?? "",
      turn_id: null,
      origin_session_id: null,
      outcome: "completed",
      error: null,
      delivery: "idle",
      notification_eligible: false,
      items: [],
    };
  }

  async remind(target: ItemTarget): Promise<boolean> {
    const request = ++this.replayRequest;
    const message = await this.item(target);
    if (request !== this.replayRequest) return false;
    this.replayBubbles.clear();
    this.replay = { target, message };
    this.replayBubbles.offer({ id: message.id, text: message.content });
    this.tick();
    this.emit();
    return true;
  }

  async explain(target: ItemTarget): Promise<void> {
    const generation = this.generation;
    const sessionId = this.model.sessionId;
    const message = await this.item(target);
    if (generation !== this.generation || sessionId !== this.model.sessionId)
      throw new UiError("itemUnavailable");
    this.state.focusTarget = null;
    this.state.focusRequest++;
    this.selectPage("remy");
    await this.send(
      translate(this.state.settings.language, "explainItemQuery", {
        content: message.content,
      }),
    );
  }

  speakMessage(id: string): void {
    const selected = this.state.selectedAgentName ?? "Remy";
    const message =
      selected === "Remy"
        ? this.replayBubbles.active?.id === id
          ? this.replay?.message
          : this.notificationMessage(id)
        : this.children
            .get(selected)
            ?.model.messages.find((message) => message.id === id);
    if (!message || message.role !== "assistant" || !message.completed)
      throw new UiError("speechNoText");
    const displayed = this.state.messages.find((item) => item.id === id);
    this.speech.speak(
      displayed
        ? { ...message, content: replyText(displayed), items: [] }
        : message,
    );
  }

  bubbleAction(
    id: string,
    action: "dismiss" | "snooze" | "pause",
    paused = false,
  ): void {
    if (this.replayBubbles.active?.id === id) {
      if (action === "snooze") return;
      if (action === "pause") this.replayBubbles.pause(id, paused, Date.now());
      else {
        this.replayBubbles.dismiss(id);
        if (this.state.speech.playback?.messageId === id) this.speech.stop();
      }
      this.tick();
      this.emit();
      return;
    }
    if (
      this.state.surface?.kind === "message" &&
      this.state.surface.message.source === "chat"
    ) {
      if (action === "snooze") return;
      if (!this.presentationController.action(id, action, paused)) return;
      if (action === "dismiss" && this.state.speech.playback?.messageId === id)
        this.speech.stop();
      this.tick();
      this.emit();
      return;
    }
    const queue =
      this.instantBubbles.active?.id === id
        ? this.instantBubbles
        : this.bubbles;
    if (id !== queue.active?.id) return;
    if (action === "pause") queue.pause(id, paused, Date.now());
    else {
      if (this.state.speech.playback?.messageId === id) this.speech.stop();
      const message = this.notificationMessage(id);
      const receipt: Receipt | undefined =
        message && this.notificationPresentation(message)?.receipts[id];
      if (receipt) {
        receipt.presented = true;
        if (action === "snooze") receipt.snoozeUntil = Date.now() + 10 * 60_000;
        this.storage.changed();
      }
      queue.dismiss(id);
    }
    this.tick();
    this.emit();
  }

  private syncAgentSelection(): void {
    const agents = this.state.agents ?? [];
    if (this.state.activePage === "tasks") {
      const tasks = agents.filter((agent) => agent.agent_type !== "chat");
      const task = tasks.find(
        (agent) => agent.agent_name === this.selectedTask,
      );
      this.activateAgent(task?.agent_name ?? tasks[0]?.agent_name ?? "Remy");
    } else if (
      this.state.activePage === "remy" ||
      !agents.some((agent) => agent.agent_name === this.state.selectedAgentName)
    ) {
      this.activateAgent("Remy");
    }
  }

  selectAgent(agentName: string): void {
    this.openAgent(agentName);
    this.emit();
  }

  private openAgent(agentName: string): void {
    if (
      !(this.state.agents ?? []).some(
        (agent) => agent.agent_name === agentName,
      ) &&
      agentName !== "Remy"
    )
      throw new UiError("invalidAction");
    this.state.activePage = agentName === "Remy" ? "remy" : "tasks";
    this.activateAgent(agentName);
  }

  private activateAgent(agentName: string): void {
    if (this.state.selectedAgentName === agentName) return;
    this.speech.cancel();
    this.state.selectedAgentName = agentName;
    this.messageInputs = [];
    this.state.focusTarget = null;
    this.childWatch?.abort();
    if (agentName !== "Remy") {
      this.selectedTask = agentName;
      if (!this.children.has(agentName))
        this.children.set(agentName, {
          model: new SessionModel(),
          pending: null,
          reply: emptyReply(),
          error: null,
          connected: false,
        });
      const controller = new AbortController();
      this.childWatch = controller;
      const signal = AbortSignal.any([
        controller.signal,
        this.lifecycle!.signal,
      ]);
      this.childWatching = this.watchChild(agentName, signal);
    }
  }

  private async watchChild(
    agentName: string,
    signal: AbortSignal,
  ): Promise<void> {
    const child = this.children.get(agentName)!;
    while (!signal.aborted) {
      try {
        await this.client().stream(
          "watch_agent_session",
          { agent_name: agentName },
          signal,
          (chunk) => {
            if (signal.aborted) return;
            if (chunk.chunk_type === "error")
              throw new Error(String(chunk.chunk));
            if (chunk.chunk_type !== "data") return;
            const previousSession = child.model.sessionId;
            child.model.apply(chunk.chunk);
            if (
              child.pending &&
              !child.pending.streaming &&
              child.model.sessionId !== previousSession
            )
              child.pending = null;
            child.connected = true;
            child.error = null;
            this.emit();
          },
        );
        if (!signal.aborted) throw new UiError("subscriptionEnded");
      } catch (error) {
        if (signal.aborted) return;
        child.connected = false;
        child.error = error;
        this.emit();
        try {
          await delay(1000, signal);
        } catch {
          return;
        }
      }
    }
  }

  private async sendChild(agentName: string, query: string): Promise<void> {
    const child = this.children.get(agentName);
    if (
      !child?.connected ||
      !child.model.sessionId ||
      !this.lifecycle ||
      !query.trim() ||
      query.length > 100000
    )
      throw new UiError("sessionUnavailable");
    if (child.pending?.streaming) throw new UiError("busy");
    const generation = this.generation;
    const client = this.client();
    const lifecycle = this.lifecycle.signal;
    this.speech.cancel();
    const pending: PendingTurn = {
      id: globalThis.crypto.randomUUID(),
      query: query.trim(),
      sessionId: child.model.sessionId,
      streaming: true,
    };
    child.pending = pending;
    child.reply = emptyReply();
    const controller = new AbortController();
    child.stream = controller;
    this.emit();
    const sending = (async () => {
      try {
        await client.stream(
          "ask_agent",
          { agent_name: agentName, query: pending.query },
          AbortSignal.any([controller.signal, lifecycle]),
          (chunk) => {
            if (generation !== this.generation) return;
            const progress = (child.reply = reduceReply(child.reply, chunk));
            Object.assign(pending, {
              turnId: progress.turn_id ?? undefined,
              outcome: progress.outcome ?? undefined,
              sessionId: progress.session_id ?? pending.sessionId,
            });
            this.emit();
          },
        );
        if (child.reply.error) throw new Error(child.reply.error);
        if (
          child.reply.outcome !== "completed" &&
          child.reply.outcome !== "waiting"
        )
          throw new UiError("replyIncomplete");
      } catch (error) {
        pending.outcome = controller.signal.aborted
          ? "interrupted"
          : (pending.outcome ?? "failed");
        pending.error = errorText(
          controller.signal.aborted ? new UiError("interrupted") : error,
          this.state.settings.language,
        );
      } finally {
        if (generation !== this.generation) return;
        pending.streaming = false;
        child.stream = undefined;
        this.emit();
      }
    })();
    child.sending = sending;
    await sending;
  }

  async newSession(): Promise<void> {
    const generation = this.generation;
    if ((this.state.selectedAgentName ?? "Remy") !== "Remy")
      throw new UiError("invalidAction");
    await this.settingsWrites;
    if (generation !== this.generation) throw new UiError("sessionUnavailable");
    if (
      !this.scope ||
      !this.lifecycle ||
      this.lifecycle.signal.aborted ||
      this.chatPending?.streaming
    )
      throw new UiError("sessionBusy");
    // Reported activity also includes reminder work. The service owns the Chat lock.
    this.speech.cancel();
    await this.client().call("create_chat_session", {}, this.lifecycle!.signal);
    if (generation !== this.generation) throw new UiError("sessionUnavailable");
    this.chatPending = null;
    this.emit();
  }

  async send(query: string): Promise<void> {
    const agentName = this.state.selectedAgentName ?? "Remy";
    if (agentName !== "Remy") return this.sendChild(agentName, query);
    const generation = this.generation;
    await this.settingsWrites;
    if (generation !== this.generation) throw new UiError("sessionUnavailable");
    query = query.trim();
    if (!query || query.length > 100_000) throw new UiError("invalidQuery");
    if (this.closed || !this.ready || this.model.status === "error")
      throw new UiError("sessionUnavailable");
    if (this.chatPending?.streaming || this.model.status === "active")
      throw new UiError("busy");
    this.speech.cancel();
    const pending: PendingTurn = {
      id: globalThis.crypto.randomUUID(),
      query,
      sessionId: this.model.sessionId,
      streaming: true,
    };
    this.chatPending = pending;
    this.state.focusTarget = null;
    this.pendingIssue = null;
    const controller = new AbortController();
    this.stream = controller;
    let progress = emptyReply();
    this.tick();
    this.emit();
    try {
      await this.client().stream(
        "ask_agent",
        { query },
        AbortSignal.any([controller.signal, this.lifecycle!.signal]),
        (chunk) => {
          if (generation !== this.generation) return;
          progress = reduceReply(progress, chunk);
          if (this.chatPending !== pending) return;
          Object.assign(pending, {
            turnId: progress.turn_id ?? undefined,
            outcome: progress.outcome ?? undefined,
            sessionId: progress.session_id ?? pending.sessionId,
          });
          this.syncPending(this.model, pending);
          this.emit();
        },
      );
      if (progress.error)
        throw typeof progress.error === "string"
          ? new Error(progress.error)
          : progress.error;
      if (progress.outcome !== "completed" && progress.outcome !== "waiting")
        throw new UiError("replyIncomplete");
    } catch (error) {
      if (generation === this.generation && this.chatPending === pending) {
        if (controller.signal.aborted) pending.outcome = "interrupted";
        this.pendingIssue = controller.signal.aborted
          ? new UiError("interrupted")
          : error;
      }
    } finally {
      if (generation === this.generation && this.chatPending === pending) {
        pending.streaming = false;
        this.stream = null;
        this.emit();
        this.reconcilePending();
        this.tick();
        this.emit();
      }
    }
  }

  private syncPending(model: SessionModel, pending: PendingTurn | null): void {
    if (!pending?.turnId) return;
    const message = this.acceptedReply(model, pending);
    if (message && (!pending.outcome || pending.outcome === "completed")) {
      pending.outcome = message.outcome ?? "completed";
      pending.error = undefined;
      if (pending === this.chatPending) this.pendingIssue = null;
    }
  }

  private reconcilePending(): void {
    const pending = this.chatPending;
    const native = this.acceptedReply(this.model, pending);
    if (!pending || !native) return;
    if (
      this.state.focusTarget?.kind === "message" &&
      this.state.focusTarget.id === `local-assistant:${pending.id}`
    ) {
      this.state.focusTarget = { kind: "message", id: native.id };
      this.state.focusRequest++;
    }
    this.chatPending = null;
  }

  private acceptedReply(
    model: SessionModel,
    pending: PendingTurn | null,
  ): AgentMessage | undefined {
    if (
      !pending?.turnId ||
      pending.streaming ||
      pending.outcome === "waiting" ||
      model.status === "active"
    )
      return;
    if (pending.error && pending.outcome && pending.outcome !== "completed")
      return;
    const replies = model.messages.filter(
      (message) =>
        message.role === "assistant" && message.turn_id === pending.turnId,
    );
    return replies.length && replies.every((message) => message.completed)
      ? replies[0]
      : undefined;
  }

  stopReply(): void {
    const selected = this.state.selectedAgentName ?? "Remy";
    if (selected !== "Remy") {
      this.children.get(selected)?.stream?.abort();
      return;
    }
    this.speech.cancel();
    this.stream?.abort();
  }

  async close(): Promise<void> {
    this.closed = true;
    this.childWatch?.abort();
    for (const child of this.children.values()) child.stream?.abort();
    clearInterval(this.clock);
    this.lifecycle?.abort();
    this.stream?.abort();
    this.generation++;
    await this.speech.close();
    await this.settingsWrites;
    await this.running;
    await this.childWatching;
    await Promise.allSettled(
      [...this.children.values()].map((child) => child.sending),
    );
    await this.storage.flush();
  }
}
