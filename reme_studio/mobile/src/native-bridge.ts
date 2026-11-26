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
          postMessage(message: {
            action: "request.open";
            id: number;
            url: string;
            method: string;
            body: string;
            accept: string;
            adminToken: string;
          }): Promise<{ status: number; contentType: string }>;
          postMessage(message: {
            action: "request.read";
            id: number;
          }): Promise<string | null>;
          postMessage(message: {
            action: "request.cancel";
            id: number;
          }): Promise<boolean>;
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
  | { action: "openExternal" | "theme" | "network.access"; value: string }
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

function decodeBase64(encoded: string): Uint8Array<ArrayBuffer> {
  return Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
}

export const iosFetch: typeof fetch | undefined =
  ios &&
  (async (url, init) => {
    if (
      typeof url !== "string" ||
      (init?.body != null && typeof init.body !== "string")
    )
      throw new TypeError("Expected a ReMe URL and optional JSON body");
    const signal = init?.signal;
    signal?.throwIfAborted();
    const id = nextRequest++;
    let stream: ReadableStreamDefaultController<Uint8Array> | undefined;
    const cleanup = () => signal?.removeEventListener("abort", abort);
    const cancel = () => {
      cleanup();
      void ios.postMessage({ action: "request.cancel", id }).catch(() => {});
    };
    const abort = () => {
      cancel();
      stream?.error(signal?.reason);
    };
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const incoming = new Headers(init?.headers);
      const { status, contentType } = await ios.postMessage({
        action: "request.open",
        id,
        url,
        method: init?.method ?? "GET",
        body: init?.body ?? "",
        accept: incoming.get("accept") ?? "",
        adminToken: incoming.get("x-reme-admin-token") ?? "",
      });
      signal?.throwIfAborted();
      const headers = contentType ? { "Content-Type": contentType } : undefined;
      if (status === 204 || status === 205 || status === 304) {
        cancel();
        return new Response(null, { status, headers });
      }
      const body = new ReadableStream<Uint8Array>(
        {
          start(controller) {
            stream = controller;
          },
          async pull(controller) {
            try {
              const chunk = await ios.postMessage({
                action: "request.read",
                id,
              });
              signal?.throwIfAborted();
              if (chunk === null) {
                cleanup();
                controller.close();
              } else controller.enqueue(decodeBase64(chunk));
            } catch (error) {
              cancel();
              throw error;
            }
          },
          cancel,
        },
        { highWaterMark: 0 },
      );
      return new Response(body, { status, headers });
    } catch (error) {
      cancel();
      signal?.throwIfAborted();
      throw error;
    }
  });

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
      const bytes = decodeBase64(stop.audio);
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
export const requestLocalNetwork = android
  ? async (url: string) => {
      const reply = await nativeRequest({
        action: "network.access",
        value: url,
      });
      if (!reply.success) throw new UiError("localNetworkDenied");
    }
  : undefined;
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
