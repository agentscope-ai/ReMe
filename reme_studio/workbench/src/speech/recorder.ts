import worklet from "./capture-worklet.js?raw";
import {
  encodeWave,
  hasAudibleSignal,
  MAX_RECORD_MS,
  SAMPLE_RATE,
} from "../../../client/src/speech.ts";
import { UiError } from "../../../client/src/i18n.ts";

const START_TIMEOUT_MS = 30_000;

export interface Recorder {
  stop(): Promise<{ audio: Uint8Array; duration: number; audible: boolean }>;
  cancel(): void;
}

export async function openRecorder(
  signal: AbortSignal,
  progress: (seconds: number, level: number) => void,
): Promise<Recorder> {
  signal.throwIfAborted();
  let stream: MediaStream | undefined;
  let context: AudioContext | undefined;
  let node: AudioWorkletNode | undefined;
  let closed = false;
  const chunks: Float32Array[] = [];
  let frames = 0;
  let resolveReady!: () => void;
  let rejectReady!: (reason: unknown) => void;
  const ready = new Promise<void>((resolve, reject) => {
    resolveReady = resolve;
    rejectReady = reject;
  });
  const cancel = () => {
    if (closed) return;
    closed = true;
    clearTimeout(timer);
    signal.removeEventListener("abort", abort);
    removeTrackListeners();
    if (node) {
      node.port.onmessage = null;
      node.port.close();
      node.disconnect();
    }
    stream?.getTracks().forEach((track) => track.stop());
    void context?.close().catch(() => {});
  };
  const fail = (reason: unknown) => {
    rejectReady(reason);
    cancel();
  };
  const abort = () => fail(signal.reason);
  const timer = setTimeout(
    () => fail(new UiError("speechMicTimeout")),
    START_TIMEOUT_MS,
  );
  let removeTrackListeners = () => {};
  signal.addEventListener("abort", abort, { once: true });
  try {
    const initialize = async () => {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new UiError("speechMicFailed");
      // Prepare processing before opening the microphone, so module loading cannot lose speech.
      context = new AudioContext({ sampleRate: SAMPLE_RATE });
      const moduleUrl = URL.createObjectURL(
        new Blob([worklet], { type: "text/javascript" }),
      );
      try {
        await context.audioWorklet.addModule(moduleUrl);
      } finally {
        URL.revokeObjectURL(moduleUrl);
      }
      if (closed) throw new UiError("speechCancelled");
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
        video: false,
      });
      // getUserMedia cannot be aborted; release a stream granted after cancellation.
      if (closed) {
        stream.getTracks().forEach((track) => track.stop());
        throw new UiError("speechCancelled");
      }
      const track = stream.getAudioTracks()[0];
      if (!track || track.readyState !== "live")
        throw new UiError("speechMicFailed");
      let hasInput = false;
      const checkReady = () => {
        if (
          hasInput &&
          context?.state === "running" &&
          track.readyState === "live" &&
          track.enabled &&
          !track.muted
        )
          resolveReady();
      };
      const ended = () => fail(new UiError("speechMicFailed"));
      track.addEventListener("ended", ended);
      removeTrackListeners = () => track.removeEventListener("ended", ended);
      node = new AudioWorkletNode(context, "reme-capture");
      const maximum = (MAX_RECORD_MS / 1000) * SAMPLE_RATE;
      node.port.onmessage = (event: MessageEvent<Float32Array>) => {
        if (closed || frames >= maximum || !event.data.length) return;
        const samples = event.data.slice(0, maximum - frames);
        chunks.push(samples);
        frames += samples.length;
        hasInput = track.enabled && !track.muted;
        checkReady();
        let blockPeak = 0;
        for (const value of samples)
          blockPeak = Math.max(blockPeak, Math.abs(value));
        progress(frames / SAMPLE_RATE, Math.min(1, blockPeak * 5));
      };
      const source = context.createMediaStreamSource(stream);
      const silent = context.createGain();
      silent.gain.value = 0;
      source.connect(node).connect(silent).connect(context.destination);
      await context.resume();
      if (closed) throw new UiError("speechCancelled");
      checkReady();
    };
    await Promise.all([initialize(), ready]);
    clearTimeout(timer);
    removeTrackListeners();
    signal.throwIfAborted();
    return {
      cancel,
      async stop() {
        await new Promise((resolve) => setTimeout(resolve, 80));
        signal.throwIfAborted();
        cancel();
        const pcm = new Int16Array(frames);
        let index = 0;
        for (const chunk of chunks)
          for (const sample of chunk) {
            const value = Math.max(-1, Math.min(1, sample));
            pcm[index++] = value * (value < 0 ? 32768 : 32767);
          }
        return {
          audio: encodeWave(pcm),
          duration: (frames / SAMPLE_RATE) * 1000,
          audible: hasAudibleSignal(pcm),
        };
      },
    };
  } catch (error) {
    cancel();
    if (signal.aborted || error instanceof UiError) throw error;
    const name = error instanceof Error ? error.name : "";
    throw new UiError(
      name === "NotAllowedError" || name === "SecurityError"
        ? "speechMicDenied"
        : name === "NotFoundError"
          ? "speechMicMissing"
          : "speechMicFailed",
    );
  }
}
