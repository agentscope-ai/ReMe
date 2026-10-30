/**
 * The LLM backend the workspace `.env` selects.
 *
 * ReMe's own configuration resolves an empty value to `openai`
 * (`${LLM_BACKEND:-openai}`), and the Host's probe resolves it the same way.
 * The tab has to agree: a backend that is only picked in the form and equals
 * the default is not a change to the configuration.
 *
 * This module stays browser-safe: no Node built-ins, because the client
 * imports it too.
 */
export const DEFAULT_LLM_BACKEND = "openai";

/** The backend a value means, with the default the configuration applies. */
export function llmBackendOf(value: string): string {
  return (value ?? "").trim().toLowerCase() || DEFAULT_LLM_BACKEND;
}
