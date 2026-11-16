import type { ToolAction } from "./protocol.ts";

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
const text = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value : undefined;

const namedActions: [RegExp, ToolAction["kind"]][] = [
  [/\b(write|edit|patch)\b|file change/, "edit"],
  [/\b(view image|image view)\b/, "image"],
  [/web extract|fetch|browse|open page/, "browse"],
  [/web search|search web/, "webSearch"],
  [/\b(search|find|grep)\b/, "search"],
  [/\blist\b/, "list"],
  [/\b(read|load)(?:$| (?:files?|directories|directory|dir|paths?)\b)/, "read"],
  [/\b(command|exec|bash|shell|terminal)\b/, "command"],
];

/** Preserve native command semantics before discarding provider-specific payloads. */
export function toolActions(name: string, value: unknown): ToolAction[] {
  const payload = record(value);
  if (Array.isArray(payload.commandActions) && payload.commandActions.length)
    return payload.commandActions.map((value) => {
      const action = record(value);
      const kind =
        action.type === "read"
          ? "read"
          : action.type === "listFiles"
            ? "list"
            : action.type === "search"
              ? "search"
              : "command";
      return {
        kind,
        target:
          text(action.path) ??
          text(kind === "command" ? action.command : action.name),
        query: text(action.query),
      };
    });
  if (
    payload.type === "fileChange" &&
    Array.isArray(payload.changes) &&
    payload.changes.length
  )
    return payload.changes.map((value) => ({
      kind: "edit",
      target: text(record(value).path),
    }));

  if (payload.type === "webSearch") {
    const action = record(payload.action);
    if (action.type === "search") {
      const queries =
        Array.isArray(action.queries) && action.queries.length
          ? action.queries
          : [action.query];
      return queries.length
        ? queries.map((query) => ({ kind: "webSearch", query: text(query) }))
        : [{ kind: "webSearch" }];
    }
    if (action.type === "openPage")
      return [{ kind: "browse", target: text(action.url) }];
    if (action.type === "findInPage")
      return [
        {
          kind: "search",
          target: text(action.url),
          query: text(action.pattern),
        },
      ];
  }

  const words = name
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_./:]+/g, " ")
    .toLowerCase();
  const kind =
    namedActions.find(([pattern]) => pattern.test(words))?.[1] ?? "tool";
  let arguments_ = payload.arguments ?? payload.input ?? payload;
  if (typeof arguments_ === "string") {
    try {
      arguments_ = JSON.parse(arguments_);
    } catch {
      /* Plain commands and partial arguments remain opaque. */
    }
  }
  const input = record(arguments_);
  const target =
    kind === "command"
      ? (text(input.command) ?? text(input.cmd) ?? text(value))
      : (text(input.path) ?? text(input.file_path) ?? text(input.url));
  return [
    {
      kind,
      target: kind === "tool" ? name : target,
      query: text(input.query) ?? text(input.pattern),
    },
  ];
}
