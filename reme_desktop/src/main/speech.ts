import { serializeError, UiError } from "../shared/i18n.ts";
import { narrationParts, validRecording } from "../shared/speech.ts";
import type {
  AgentActivity,
  DesktopCommand,
  SessionMessage,
  Settings,
  SpeechAudio,
  SpeechState,
} from "../shared/protocol.ts";
import type { ReMeClient } from "./reme-client.ts";

export const emptySpeech = (): SpeechState => ({
  asr: false,
  tts: false,
  ready: false,
  capture: null,
  playback: null,
  error: null,
});
interface Context {
  sessionId: string | null;
  connected: boolean;
  chatVisible: boolean;
  visible: boolean;
  settings: Settings;
}
interface Playback {
  id: number;
  controller: AbortController;
  parts: string[];
  index: number;
  deadline: number;
  audio: Map<number, SpeechAudio>;
  producing: boolean;
  primed: boolean;
  sent: boolean;
  failure?: { index: number; error: unknown };
  turnId?: string | null;
  immediate: boolean;
}

/** One capture or narration at a time. Audio never enters history or state broadcasts. */
export class SpeechController {
  state = emptySpeech();
  private rendererToken: string | undefined;
  private sequence = 0;
  private capture: {
    id: string;
    controller: AbortController;
    deadline: number;
  } | null = null;
  private playback: Playback | null = null;
  private readonly seen = new Set<string>();
  private readonly candidates = new Map<string, SessionMessage>();
  private readonly tasks = new Set<Promise<void>>();
  private readonly cache = new Map<
    string,
    { audio: SpeechAudio; size: number; until: number }
  >();
  private cacheBytes = 0;
  private namespace = "";
  private closed = false;

  constructor(
    private readonly context: () => Context,
    private readonly client: () => ReMeClient,
    private readonly publish: (state: SpeechState) => void,
    private readonly command: (command: DesktopCommand) => void,
  ) {}

  private update(patch: Partial<SpeechState>): void {
    this.state = { ...this.state, ...patch };
    this.publish(this.state);
  }

  configure(asr: boolean, tts: boolean, namespace: string): void {
    if (!asr) this.cancelCapture();
    if (!tts) this.stop();
    if (namespace !== this.namespace) {
      this.cancel();
      this.cache.clear();
      this.cacheBytes = 0;
      this.namespace = namespace;
    }
    this.update({ asr, tts });
  }

  rendererReady(ready: boolean, token?: string): void {
    if (this.closed) return;
    if (!ready && token !== undefined && token !== this.rendererToken) return;
    if (
      ready &&
      this.rendererToken !== undefined &&
      token !== this.rendererToken
    )
      this.cancel();
    this.rendererToken = ready ? token : undefined;
    this.update({ ready });
    if (!ready) this.cancel();
  }

  reset(): void {
    this.cancel();
    this.seen.clear();
    this.candidates.clear();
    this.clearError();
  }

  clearError(): void {
    if (this.state.error) this.update({ error: null });
  }

  baseline(messages: SessionMessage[]): void {
    this.candidates.clear();
    for (const message of messages)
      if (message.completed) this.seen.add(message.id);
  }

  observe(message: SessionMessage): void {
    if (
      message.role !== "assistant" ||
      message.source !== "chat" ||
      !message.completed ||
      this.seen.has(message.id)
    )
      return;
    this.seen.add(message.id);
    this.candidates.set(message.id, message);
  }

  finish(activity: AgentActivity | undefined): void {
    if (!activity) return;
    if (!["completed", "failed", "interrupted"].includes(activity.phase)) {
      if (
        this.playback &&
        !this.playback.immediate &&
        this.state.playback?.kind !== "manual" &&
        activity.turn_id !== this.playback.turnId
      )
        this.stop();
      return;
    }
    for (const [id, message] of this.candidates) {
      if (message.turn_id !== activity.turn_id) continue;
      this.candidates.delete(id);
      if (
        activity.phase === "completed" &&
        this.context().settings.voiceReplies
      )
        this.speak(message, "reply");
    }
  }

  private automaticAllowed(immediate = false): boolean {
    const context = this.context();
    return (
      !this.closed &&
      this.state.ready &&
      context.connected &&
      context.visible &&
      context.settings.voiceEnabled &&
      (immediate || context.settings.quietUntil <= Date.now()) &&
      context.settings.voiceVolume > 0
    );
  }

  speak(
    message: SessionMessage,
    kind: NonNullable<SpeechState["playback"]>["kind"] = "manual",
  ): void {
    if (
      kind !== "manual" &&
      (!this.automaticAllowed(
        kind === "reminder" && message.delivery === "immediate",
      ) ||
        this.state.capture ||
        this.state.playback)
    )
      return;
    if (kind === "reminder" && !this.context().settings.voiceReminders) return;
    if (!this.state.tts) {
      if (kind === "manual") throw new UiError("speechTtsUnavailable");
      return;
    }
    if (this.closed || !this.context().connected)
      throw new UiError("sessionUnavailable");
    if (!this.state.ready) throw new UiError("speechNotReady");
    const parts = narrationParts(message.content);
    if (!parts.length) {
      if (kind === "manual") throw new UiError("speechNoText");
      return;
    }
    this.cancel();
    const playback: Playback = {
      id: ++this.sequence,
      controller: new AbortController(),
      parts,
      index: 0,
      deadline: Infinity,
      audio: new Map(),
      producing: false,
      primed: false,
      sent: false,
      turnId: message.turn_id,
      immediate: message.delivery === "immediate",
    };
    this.playback = playback;
    this.update({
      playback: {
        id: playback.id,
        messageId: message.id,
        kind,
        phase: "loading",
        part: 1,
        total: parts.length,
      },
      error: null,
    });
    this.fill(playback);
  }

  private active(playback: Playback): boolean {
    return this.playback === playback && !playback.controller.signal.aborted;
  }

  /** A single producer fills at most two future blocks; it never owns playback order. */
  private fill(playback: Playback): void {
    if (!this.active(playback) || playback.producing || playback.failure)
      return;
    playback.producing = true;
    const task = (async () => {
      while (this.active(playback) && !playback.failure) {
        const end = Math.min(
          playback.parts.length,
          playback.index + (playback.sent ? 3 : 2),
        );
        let index = playback.index;
        while (index < end && playback.audio.has(index)) index++;
        if (index === end) return;
        if (!playback.sent && this.state.playback)
          this.update({
            playback: { ...this.state.playback, preparing: index + 1 },
          });
        for (let attempt = 0; attempt < 2; attempt++) {
          try {
            const audio = await this.loadAudio(playback, playback.parts[index]);
            if (!this.active(playback)) return;
            if (audio) playback.audio.set(index, audio);
            break;
          } catch (error) {
            if (!this.active(playback)) return;
            if (attempt === 1) playback.failure = { index, error };
          }
        }
        this.advance(playback);
      }
    })()
      .catch((error) => {
        if (this.active(playback)) {
          playback.failure = { index: playback.index, error };
          this.advance(playback);
        }
      })
      .finally(() => {
        playback.producing = false;
      });
    this.tasks.add(task);
    void task.finally(() => this.tasks.delete(task));
  }

  private advance(playback: Playback): void {
    if (!this.active(playback) || playback.sent || !this.state.playback) return;
    const audio = playback.audio.get(playback.index);
    if (!audio) {
      if (playback.failure?.index === playback.index)
        this.suspend(playback, playback.failure.error);
      return;
    }
    if (
      !playback.primed &&
      !playback.failure &&
      playback.audio.size < Math.min(2, playback.parts.length)
    )
      return;
    playback.primed = true;
    playback.sent = true;
    playback.deadline = Date.now() + 10_000;
    this.update({
      playback: {
        ...this.state.playback,
        phase: "loading",
        preparing: undefined,
      },
    });
    this.command({ type: "speech.play", id: playback.id, audio });
  }

  private suspend(playback: Playback, error: unknown): void {
    playback.sent = false;
    playback.deadline = Infinity;
    playback.failure = { index: playback.index, error };
    this.command({ type: "speech.stop", id: playback.id });
    this.update({
      playback: {
        ...this.state.playback!,
        phase: "error",
        preparing: undefined,
      },
      error: serializeError(error),
    });
  }

  retry(): void {
    const playback = this.playback;
    if (!playback || this.state.playback?.phase !== "error") return;
    if (this.closed || !this.context().connected)
      throw new UiError("sessionUnavailable");
    if (!this.state.ready) throw new UiError("speechNotReady");
    playback.failure = undefined;
    playback.id = ++this.sequence;
    this.update({
      playback: {
        ...this.state.playback,
        id: playback.id,
        kind: "manual",
        phase: "loading",
      },
      error: null,
    });
    this.advance(playback);
    this.fill(playback);
  }

  private async loadAudio(
    playback: Playback,
    text: string,
  ): Promise<SpeechAudio | undefined> {
    const key = `${this.namespace}\0${text}`;
    const cached = this.cache.get(key);
    let audio: SpeechAudio;
    if (cached && cached.until > Date.now()) {
      audio = cached.audio;
      this.cache.delete(key);
      this.cache.set(key, cached);
    } else {
      const result = await this.client().call<
        SpeechAudio & { is_last: boolean }
      >("tts_demo", { text }, playback.controller.signal, {
        timeoutMs: 70_000,
        maxBytes: 16 * 1024 * 1024,
      });
      if (this.playback !== playback || playback.controller.signal.aborted)
        return;
      if (
        !result ||
        result.is_last !== true ||
        typeof result.audio !== "string" ||
        !/^[A-Za-z0-9+/]+={0,2}$/.test(result.audio) ||
        result.audio.length % 4 !== 0 ||
        !["wav", "mp3", "pcm", "opus"].includes(result.format) ||
        !Number.isInteger(result.sample_rate) ||
        result.sample_rate < 8000 ||
        result.sample_rate > 192000
      )
        throw new UiError("speechInvalidAudio");
      audio = {
        audio: result.audio,
        format: result.format,
        sample_rate: result.sample_rate,
      };
      if (cached) {
        this.cacheBytes -= cached.size;
        this.cache.delete(key);
      }
      const size = audio.audio.length;
      this.cache.set(key, { audio, size, until: Date.now() + 5 * 60_000 });
      this.cacheBytes += size;
      while (this.cache.size > 40 || this.cacheBytes > 24 * 1024 * 1024) {
        const first = this.cache.keys().next().value!;
        this.cacheBytes -= this.cache.get(first)!.size;
        this.cache.delete(first);
      }
    }
    return this.active(playback) ? audio : undefined;
  }

  report(id: number, status: "started" | "ended" | "failed"): void {
    const playback = this.playback;
    if (
      !playback ||
      id !== playback.id ||
      !this.state.playback ||
      !playback.sent
    )
      return;
    if (status === "started") {
      playback.deadline = Date.now() + 180_000;
      this.update({ playback: { ...this.state.playback, phase: "playing" } });
    } else if (status === "ended" && this.state.playback.phase === "playing") {
      playback.audio.delete(playback.index);
      playback.sent = false;
      playback.deadline = Infinity;
      if (++playback.index < playback.parts.length) {
        playback.id = ++this.sequence;
        this.update({
          playback: {
            ...this.state.playback,
            id: playback.id,
            phase: "loading",
            part: playback.index + 1,
          },
        });
        this.advance(playback);
        this.fill(playback);
      } else this.stop();
    } else if (status === "failed") {
      playback.audio.delete(playback.index);
      const key = `${this.namespace}\0${playback.parts[playback.index]}`;
      const cached = this.cache.get(key);
      if (cached) {
        this.cacheBytes -= cached.size;
        this.cache.delete(key);
      }
      this.suspend(playback, new UiError("speechPlaybackFailed"));
    }
  }

  beginCapture(id: string): void {
    if (!this.state.asr) throw new UiError("speechAsrUnavailable");
    if (
      this.closed ||
      !this.context().connected ||
      !this.context().chatVisible ||
      !this.state.ready
    )
      throw new UiError("speechNotReady");
    if (typeof id !== "string" || !id || id.length > 80)
      throw new UiError("invalidAction");
    this.cancel();
    this.capture = {
      id,
      controller: new AbortController(),
      deadline: Date.now() + 60_000,
    };
    this.update({ capture: { id, phase: "starting" }, error: null });
  }

  recording(id: string): void {
    if (this.capture?.id !== id || this.state.capture?.phase !== "starting")
      throw new UiError("speechCancelled");
    this.capture.deadline = Date.now() + MAX_CAPTURE_WAIT;
    this.update({ capture: { id, phase: "recording" } });
  }

  async transcribe(id: string, audio: Uint8Array): Promise<string> {
    const capture = this.capture;
    if (
      !capture ||
      capture.id !== id ||
      this.state.capture?.phase !== "recording"
    )
      throw new UiError("speechCancelled");
    if (!(audio instanceof Uint8Array) || !validRecording(audio)) {
      this.cancelCapture(id);
      throw new UiError("speechInvalidRecording");
    }
    capture.deadline = Date.now() + 75_000;
    this.update({ capture: { id, phase: "transcribing" } });
    try {
      const text = await this.client().call<string>(
        "asr_demo",
        {
          audio: Buffer.from(audio).toString("base64"),
          audio_format: "wav",
          sample_rate: 16_000,
        },
        capture.controller.signal,
        { timeoutMs: 70_000, maxBytes: 1024 * 1024 },
      );
      if (this.capture !== capture || capture.controller.signal.aborted)
        throw new UiError("speechCancelled");
      if (typeof text !== "string")
        throw new UiError("speechInvalidTranscript");
      return text.trim();
    } finally {
      if (this.capture === capture) this.cancelCapture(id);
    }
  }

  cancelCapture(id?: string): void {
    if (!this.capture || (id !== undefined && this.capture.id !== id)) return;
    this.capture.controller.abort();
    this.capture = null;
    this.update({ capture: null });
  }

  stop(): void {
    const playback = this.playback;
    if (!playback) return;
    this.playback = null;
    playback.controller.abort();
    this.command({ type: "speech.stop", id: playback.id });
    this.update({ playback: null });
  }

  cancel(): void {
    this.cancelCapture();
    this.stop();
  }

  tick(now = Date.now()): void {
    if (
      this.capture &&
      (!this.context().chatVisible ||
        !this.context().connected ||
        now > this.capture.deadline)
    )
      this.cancelCapture();
    if (!this.playback) return;
    if (
      !this.context().connected ||
      (this.state.playback?.kind !== "manual" &&
        (!this.automaticAllowed(this.playback.immediate) ||
          (this.state.playback?.kind === "reply" &&
            !this.context().settings.voiceReplies) ||
          (this.state.playback?.kind === "reminder" &&
            !this.context().settings.voiceReminders)))
    )
      this.stop();
    else if (now > this.playback.deadline) {
      this.suspend(this.playback, new UiError("speechPlaybackFailed"));
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    this.cancel();
    await Promise.allSettled(this.tasks);
    this.cache.clear();
  }
}

const MAX_CAPTURE_WAIT = 30_000;
