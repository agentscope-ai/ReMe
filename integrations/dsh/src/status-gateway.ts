import type { Context } from "@deepseek-ai/cordis";
import { Remote, TypertRemoteService } from "@deepseek-ai/dsh-typert-protocol";

import type { ReMeRuntime } from "./runtime.js";
import type { ReMeRuntimeSnapshot } from "./runtime-status.js";

declare module "@deepseek-ai/cordis" {
  interface Context {
    /** Active ReMe integration runtime mounted by the DSH adapter. */
    remeMemory: ReMeRuntime;
  }
}

/** Read-only Host projection consumed by the local ReMe status page. */
export class ReMeStatusGateway extends TypertRemoteService {
  static inject = ["remeMemory"];

  constructor(ctx: Context) {
    super(ctx, "remeStatus");
  }

  /** Return current queue, task, and Auto Dream scheduling state. */
  @Remote("runtime")
  runtime(): ReMeRuntimeSnapshot {
    return this.ctx.remeMemory.snapshot();
  }
}

export default ReMeStatusGateway;
