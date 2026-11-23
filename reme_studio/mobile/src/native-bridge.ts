import { UiError } from "../../client/src/i18n.ts";
import {
  encodeWave,
  hasAudibleSignal,
  SAMPLE_RATE,
} from "../../client/src/speech.ts";
import type { Recorder } from "../../workbench/src/speech/recorder.ts";

declare global {
  interface Window {
    RemeAndroid?: {
      postMessage(message: string): void;
      onmessage: ((event: MessageEvent<string>) => void) | null;
    };
    webkit?: {
      messageHandlers?: {
        remeStudio?: {
          postMessage(message: NativeMessage): Promise<NativeReply>;
        };
      };
    };
    remeHandleBack?: () => boolean;
    remeSetWindowInsets?: (
      top: number,
      right: number,
      bottom: number,
      left: number,
      backGesture: number,
      keyboard: boolean,
    ) => void;
  }
}

const android = window.RemeAndroid;
const ios = window.webkit?.messageHandlers?.remeStudio;
type NativeMessage =
  | { action: "openExternal" | "theme"; value: string }
  | {
      action:
        | "recording.start"
        | "recording.stop"
        | "recording.progress"
        | "recording.cancel";
    };
type NativeReply = {
  success: boolean;
  denied?: boolean;
  audio?: string;
  seconds?: number;
  level?: number;
};
let nextRequest = 0;
const replies = new Map<number, (reply: NativeReply) => void>();
if (android) {
  android.onmessage = ({ data }) => {
    const reply = JSON.parse(data) as NativeReply & { id: number };
    replies.get(reply.id)?.(reply);
  };
}

async function openAndroidRecorder(
  signal: AbortSignal,
  progress: (seconds: number, level: number) => void,
): Promise<Recorder> {
  signal.throwIfAborted();
  let polling: ReturnType<typeof setInterval> | undefined;
  let closed = false;
  const dispose = () => {
    closed = true;
    clearInterval(polling);
    signal.removeEventListener("abort", cancel);
  };
  const cancel = () => {
    if (closed) return;
    dispose();
    void nativeRequest({ action: "recording.cancel" });
  };
  signal.addEventListener("abort", cancel, { once: true });
  const response = await nativeRequest({ action: "recording.start" });
  signal.throwIfAborted();
  if (!response.success) {
    cancel();
    throw new UiError(response.denied ? "speechMicDenied" : "speechMicFailed");
  }
  polling = setInterval(async () => {
    const state = await nativeRequest({ action: "recording.progress" });
    if (!closed && state.success)
      progress(state.seconds ?? 0, state.level ?? 0);
  }, 100);
  return {
    cancel,
    async stop() {
      dispose();
      const stop = await nativeRequest({ action: "recording.stop" });
      signal.throwIfAborted();
      if (!stop.success || typeof stop.audio !== "string")
        throw new UiError("speechMicFailed");
      const bytes = Uint8Array.from(atob(stop.audio), (char) =>
        char.charCodeAt(0),
      );
      const view = new DataView(bytes.buffer);
      const pcm = new Int16Array(bytes.length / 2);
      for (let index = 0; index < pcm.length; index++)
        pcm[index] = view.getInt16(index * 2, true);
      return {
        audio: encodeWave(pcm),
        duration: (pcm.length * 1000) / SAMPLE_RATE,
        audible: hasAudibleSignal(pcm),
      };
    },
  };
}

export function nativeRequest(message: NativeMessage): Promise<NativeReply> {
  if (ios) return ios.postMessage(message).catch(() => ({ success: false }));
  if (!android) return Promise.resolve({ success: false });
  return new Promise((resolve) => {
    const id = nextRequest++;
    const timeout = setTimeout(() => {
      replies.delete(id);
      resolve({ success: false });
    }, 60_000);
    replies.set(id, (reply) => {
      clearTimeout(timeout);
      replies.delete(id);
      resolve(reply);
    });
    android.postMessage(JSON.stringify({ id, ...message }));
  });
}

export const platform = ios ? "ios" : "android";
export const openNativeRecorder = android ? openAndroidRecorder : undefined;
if (android) {
  const root = document.documentElement;
  root.dataset.androidInsets = "pending";
  window.remeSetWindowInsets = (
    top,
    right,
    bottom,
    left,
    backGesture,
    keyboard,
  ) => {
    root.style.setProperty("--mobile-safe-top", `${top}px`);
    root.style.setProperty("--mobile-safe-right", `${right}px`);
    root.style.setProperty("--mobile-safe-bottom", `${bottom}px`);
    root.style.setProperty("--mobile-safe-left", `${left}px`);
    root.style.setProperty("--mobile-back-gesture", `${backGesture}px`);
    root.dataset.keyboard = String(keyboard);
    root.dataset.androidInsets = "ready";
  };
}
