import { UiError } from "./i18n.ts";
import { readStream } from "./stream.ts";
import type { JobResponse, StreamChunk } from "./protocol.ts";
import type { BackendRequest } from "./backend.ts";

export class ReMeClient {
  constructor(
    readonly endpoint: string,
    private readonly fetchRequest: typeof fetch = (...args) =>
      globalThis.fetch(...args),
  ) {}

  async call<T>(
    job: string,
    body: object,
    signal: AbortSignal,
    options: { timeoutMs?: number; maxBytes?: number } = {},
  ): Promise<T> {
    return (await this.request<T>(job, body, signal, options)).answer;
  }

  async request<T>(
    job: string,
    body: object,
    signal: AbortSignal,
    options: { timeoutMs?: number; maxBytes?: number } = {},
  ): Promise<JobResponse<T>> {
    const response = await this.fetchRequest(`${this.endpoint}/${job}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.any([
        signal,
        AbortSignal.timeout(options.timeoutMs ?? 30_000),
      ]),
    });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      throw new UiError("httpError", { job, status: response.status });
    }
    let payload: JobResponse<T>;
    if (options.maxBytes && response.body) {
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let size = 0;
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          size += value.byteLength;
          if (size > options.maxBytes) throw new UiError("responseTooLarge");
          chunks.push(value);
        }
        const decoder = new TextDecoder();
        const json = chunks
          .map((chunk) => decoder.decode(chunk, { stream: true }))
          .join("");
        payload = JSON.parse(json + decoder.decode());
      } finally {
        await reader.cancel().catch(() => {});
        reader.releaseLock();
      }
    } else payload = await response.json();
    if (payload?.success !== true) {
      if (job === "save_file" && payload?.metadata?.code === "file_conflict")
        throw new UiError("documentConflict");
      throw typeof payload?.answer === "string"
        ? new Error(payload.answer)
        : new UiError("jobFailed", { job });
    }
    return { ...payload, metadata: payload.metadata ?? {} };
  }

  async manage(request: BackendRequest, signal: AbortSignal): Promise<unknown> {
    const { action } = request;
    const endpoint = new URL(this.endpoint);
    if (
      endpoint.protocol !== "https:" &&
      !["localhost", "127.0.0.1", "[::1]"].includes(endpoint.hostname)
    )
      throw new UiError("backendHttpsRequired");
    if (
      !["config", "status", "apply", "restart", "test_model"].includes(action)
    )
      throw new UiError("invalidWorkspaceRequest");
    const query =
      action === "status" && request.operationId
        ? `?${new URLSearchParams({ operation_id: request.operationId })}`
        : "";
    const headers = new Headers({ "Content-Type": "application/json" });
    if (request.credentials?.endpoint === this.endpoint)
      headers.set("X-ReMe-Admin-Token", request.credentials.token);
    const response = await this.fetchRequest(
      `${this.endpoint}/admin/${action}${query}`,
      {
        method: action === "config" || action === "status" ? "GET" : "POST",
        headers,
        body:
          "settings" in request ? JSON.stringify(request.settings) : undefined,
        redirect: "error",
        signal: AbortSignal.any([signal, AbortSignal.timeout(130_000)]),
      },
    );
    if (response.status === 401) {
      await response.body?.cancel().catch(() => {});
      throw new UiError("backendAuthRequired");
    }
    const payload = await response.json().catch(() => null);
    if (!response.ok)
      throw typeof payload?.detail === "string"
        ? new Error(payload.detail)
        : new UiError("httpError", { job: action, status: response.status });
    return payload;
  }

  async stream(
    job: "watch_agent_session" | "watch_proactive_overview" | "ask_agent",
    body: object,
    signal: AbortSignal,
    onChunk: (chunk: StreamChunk) => void,
  ): Promise<void> {
    const idle = new AbortController();
    const joined = AbortSignal.any([signal, idle.signal]);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const touch = () => {
      // Agent subscriptions promise heartbeats; a chat can wait on a long tool.
      if (job === "ask_agent") return;
      clearTimeout(timer);
      timer = setTimeout(
        () => idle.abort(new UiError("idleConnection")),
        60_000,
      );
    };
    touch();
    try {
      const response = await this.fetchRequest(`${this.endpoint}/${job}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
        signal: joined,
      });
      if (!response.ok || !response.body) {
        await response.body?.cancel().catch(() => {});
        throw new UiError("httpError", { job, status: response.status });
      }
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
