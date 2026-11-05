/**
 * The LLM backend the workspace `.env` selects.
 *
 * ReMe applies its default (`${LLM_BACKEND:-openai}`) only when the variable
 * is **absent**: an explicitly empty value stays empty and names no backend,
 * and the service then refuses to start. So the default has to be written out
 * rather than left blank — see `environment/env-fields.ts`, which owns that
 * rule.
 *
 * `llmBackendOf` is the *form's* reading, not the service's: an empty
 * selection there means "use the default", so it resolves to the default
 * backend. That is what the tab compares with, so picking the default in the
 * form is not a change to the configuration.
 *
 * This module stays browser-safe: no Node built-ins, because the client
 * imports it too.
 */
export const DEFAULT_LLM_BACKEND = "openai";

/** The backend a value means in the form, with the default the configuration applies. */
export function llmBackendOf(value: string): string {
  return (value ?? "").trim().toLowerCase() || DEFAULT_LLM_BACKEND;
}
