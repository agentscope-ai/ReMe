import type { OpenClawPluginApi } from "openclaw/plugin-sdk/plugin-entry";
import { Type } from "typebox";

import type { ReMeClientLike } from "../core/types.js";
import type { OpenClawReMeConfig } from "./config.js";

/** Register the explicit search action advertised by the plugin manifest. */
export function registerOpenClawTools(
  api: Pick<OpenClawPluginApi, "registerTool">,
  client: Pick<ReMeClientLike, "search">,
  config: Pick<OpenClawReMeConfig, "searchLimit" | "recallMinScore">,
): void {
  api.registerTool(
    {
      name: "reme_search",
      label: "ReMe Search",
      description:
        "Search ReMe long-term memory for relevant historical context.",
      parameters: Type.Object({
        query: Type.String({ description: "Focused memory search query" }),
        limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 50 })),
        min_score: Type.Optional(Type.Number({ minimum: 0 })),
      }),
      async execute(_toolCallId, params) {
        const input = params as {
          query: string;
          limit?: number;
          min_score?: number;
        };
        const query = String(input.query || "").trim();
        if (!query) return toolResult("Error: query cannot be empty.", false);
        const result = await client.search(query, {
          limit: clamp(input.limit, 1, 50, config.searchLimit),
          minScore: minimumScore(input.min_score, config.recallMinScore),
        });
        if (!result.ok)
          return toolResult(
            `ReMe search failed: ${result.error || "unknown error"}`,
            false,
          );
        const answer =
          typeof result.answer === "string"
            ? result.answer.trim()
            : JSON.stringify(result.answer, null, 2);
        return toolResult(answer || "No relevant memory found.", true);
      },
    },
    { name: "reme_search" },
  );
}

function toolResult(text: string, ok: boolean) {
  return { content: [{ type: "text" as const, text }], details: { ok, text } };
}

function clamp(
  value: unknown,
  minimum: number,
  maximum: number,
  fallback: number,
): number {
  const number = Math.round(Number(value));
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(maximum, number));
}

function minimumScore(value: unknown, fallback: number): number {
  if (value === undefined) return fallback;
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, number) : fallback;
}
