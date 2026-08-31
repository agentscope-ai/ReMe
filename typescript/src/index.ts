export { ReMeClient } from "./core/client.js";
export { formatReMeContext } from "./core/context.js";

/** Empty Host face used to compose this package's DSH browser bundle. */
export function apply(): void {}
export type {
  AutoMemoryOptions,
  DreamOptions,
  LoggerLike,
  ReMeClientConfig,
  ReMeClientLike,
  ReMeComponentHealth,
  ReMeComponentMemory,
  ReMeHealth,
  ReMeHealthResult,
  ReMeMemoryStatus,
  ReMeMessage,
  ReMeResult,
  ReMeStatusResult,
  SearchOptions,
} from "./core/types.js";
