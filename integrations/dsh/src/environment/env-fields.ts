/**
 * The `.env` keys the Environment tab manages, and the rules for updating them.
 *
 * This module is the single source of truth for the managed keys, for which of
 * them are write-only secrets, and for what an empty draft means. The Host and
 * the browser both derive their behaviour from it, so the two can never drift
 * apart about the same file.
 *
 * It stays browser-safe: no node built-in is imported here, and the browser
 * bundle must stay loadable without one.
 */

/** What an empty draft for one managed key means. */
export type EnvEmptyRule =
  /**
   * "Leave blank to keep": an empty draft writes nothing, so the stored value
   * survives. A credential is write-only — the stored value never rides a
   * response — so emptying the control cannot mean "delete my key".
   */
  | "keep"
  /**
   * An explicit clear: an empty draft writes an empty value, so the key is
   * removed from the effective configuration.
   */
  | "clear";

export interface EnvFieldMeta {
  /** The `.env` key. */
  key: string;
  /** True for write-only credentials; the stored value never rides a response. */
  secret: boolean;
  /** What an empty draft means for this key. */
  empty: EnvEmptyRule;
  /**
   * Value written when the key is unset and must not stay empty.
   *
   * ReMe 0.4.0.x instantiates the embedding component unconditionally, so
   * `EMBEDDING_API_KEY` has to hold a non-empty value even when vector search
   * is off; `EMPTY` is the documented placeholder for "no embedding key".
   */
  disabledValue?: string;
}

/**
 * The managed keys, in the order the two form groups show them.
 *
 * The order is the order the summary reports them in, so the Host projection
 * and the browser form list the same keys in the same order.
 */
export const ENV_FIELDS: readonly EnvFieldMeta[] = [
  { key: "LLM_BACKEND", secret: false, empty: "clear" },
  { key: "LLM_MODEL_NAME", secret: false, empty: "clear" },
  { key: "LLM_API_KEY", secret: true, empty: "keep" },
  { key: "LLM_BASE_URL", secret: false, empty: "clear" },
  { key: "EMBEDDING_BACKEND", secret: false, empty: "clear" },
  { key: "EMBEDDING_MODEL_NAME", secret: false, empty: "clear" },
  {
    key: "EMBEDDING_API_KEY",
    secret: true,
    empty: "keep",
    disabledValue: "EMPTY",
  },
  { key: "EMBEDDING_BASE_URL", secret: false, empty: "clear" },
];

/** Every key the Environment tab manages. */
export const MANAGED_KEYS: readonly string[] = ENV_FIELDS.map(
  (field) => field.key,
);

/** The keys whose stored value is never returned to the browser. */
export const SECRET_KEYS: ReadonlySet<string> = new Set(
  ENV_FIELDS.filter((field) => field.secret).map((field) => field.key),
);

/** The metadata of one managed key. */
export function envFieldMeta(key: string): EnvFieldMeta | undefined {
  return ENV_FIELDS.find((field) => field.key === key);
}

/**
 * Whether the file already holds a non-empty value for each managed key.
 *
 * The Host builds it from the parsed file; the tab builds it from the masked
 * projection it received, which never carries a secret's value.
 */
export function storedState(
  values: Readonly<Record<string, string>>,
): Record<string, boolean> {
  const state: Record<string, boolean> = {};
  for (const field of ENV_FIELDS)
    state[field.key] = (values[field.key] ?? "") !== "";
  return state;
}

/**
 * The update a draft of the managed keys produces.
 *
 * Only touched keys travel: an untouched key keeps its stored value. An empty
 * secret draft is dropped rather than written, so "leave blank to keep" holds
 * even after the user typed a replacement and cleared it again. An empty
 * non-secret draft is an explicit clear and keeps that meaning.
 *
 * `stored` reports, per key, whether the file already holds a non-empty value.
 * It is what decides whether a key that must not stay empty gets its
 * `disabledValue` marker.
 */
export function envUpdates(
  draft: Readonly<Record<string, string>>,
  stored: Readonly<Record<string, boolean>>,
): Record<string, string> {
  const updates: Record<string, string> = {};
  // A key the tab does not manage travels unchanged: the Host must not
  // silently drop a value it was asked to write.
  for (const [key, value] of Object.entries(draft))
    if (!envFieldMeta(key)) updates[key] = value;
  for (const field of ENV_FIELDS) {
    const typed = draft[field.key];
    const isSet = stored[field.key] === true;
    if (typed === undefined) {
      // Untouched: the stored value survives. The one exception is a key that
      // must not stay empty, which gets its documented placeholder.
      if (field.disabledValue && !isSet)
        updates[field.key] = field.disabledValue;
      continue;
    }
    if (typed === "") {
      if (field.empty === "keep") {
        // "Leave blank to keep": the stored credential survives.
        if (field.disabledValue && !isSet)
          updates[field.key] = field.disabledValue;
        continue;
      }
      updates[field.key] = "";
      continue;
    }
    updates[field.key] = typed;
  }
  return updates;
}
