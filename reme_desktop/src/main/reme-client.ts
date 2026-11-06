import { UiError } from "../shared/i18n.ts";
import { readStream } from "../shared/stream.ts";
import type { StreamChunk } from "../shared/protocol.ts";

export class ReMeClient {
  constructor(readonly endpoint: string) {}

  async call<T>(job: string, body: object, signal: AbortSignal): Promise<T> {
    const response = await fetch(`${this.endpoint}/${job}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
    });
    if (!response.ok)
      throw new UiError("httpError", { job, status: response.status });
    const payload = (await response.json()) as { success: boolean; answer: T };
    if (payload.success !== true)
      throw typeof payload.answer === "string"
        ? new Error(payload.answer)
        : new UiError("jobFailed", { job });
    return payload.answer;
  }

  async stream(
    job: "pa_events" | "pa_chat",
    body: object,
    signal: AbortSignal,
    onChunk: (chunk: StreamChunk) => void,
  ): Promise<void> {
    const idle = new AbortController();
    const joined = AbortSignal.any([signal, idle.signal]);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const touch = () => {
      // PA subscriptions promise heartbeats; a chat can wait on a long tool.
      if (job !== "pa_events") return;
      clearTimeout(timer);
      timer = setTimeout(
        () => idle.abort(new UiError("idleConnection")),
        60_000,
      );
    };
    touch();
    try {
      const response = await fetch(`${this.endpoint}/${job}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: joined,
      });
      if (!response.ok || !response.body)
        throw new UiError("httpError", { job, status: response.status });
      if (
        response.headers.get("content-type")?.split(";")[0].trim() !==
        "text/event-stream"
      ) {
        await response.body.cancel();
        throw new UiError("expectedStream", { job });
      }
      await readStream(response.body, (chunk) => {
        touch();
        onChunk(chunk);
      });
    } finally {
      clearTimeout(timer);
    }
  }
}

export async function delay(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw signal.reason;
  await new Promise<void>((resolve, reject) => {
    const abort = () => {
      clearTimeout(timer);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal.addEventListener("abort", abort, { once: true });
  });
}
