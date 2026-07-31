---
name: reme-memory
description: Use ReMe as file-native long-term memory in Claude Code. RECALL — search ReMe before answering questions about past conversations, preferences, project history, or decisions.
---

# ReMe Memory

ReMe is the persistent, file-native memory layer for this agent. It stores conversations and
resources as Markdown files with frontmatter and `[[wikilinks]]`, and consolidates them into
long-term **digest** knowledge. Your job with this plugin is **recall**.

The recall tools come from the `reme` MCP server (surfaced as `mcp__reme__…`): `search`, `traverse`,
`daily_list`, `frontmatter_read`, `read`. They are only available when the user has the server
running:

```
reme start service.backend=mcp service.transport=streamable-http
```

If the tools are missing, that server is not running — tell the user the command above instead of
guessing answers.

## Recall (read long-term memory)

Before answering questions about previous conversations, user preferences, project history,
decisions, or long-term context, recall from ReMe first. ReMe answers **three independent kinds of
question** — pick the mode the request needs; don't merge them into one call. Durable knowledge
lives under `digest/`, daily notes under `daily/`, external materials under `resource/`.

1. **Semantic** (default — "what do we know about X?"): `search` with `query="<question/keywords>"`,
   `limit=5` (optional `min_score`). Hybrid vector + BM25 with one-hop wikilink expansion.
2. **Topological** ("what links to this node?"): `traverse` with `path="<node>"`, `depth=1`
   (raise to 2 only when needed), `direction=both` to walk the `[[wikilink]]` graph.
3. **State** ("what exists / what was recorded on <date>?"): `daily_list` with `date="YYYY-MM-DD"`
   (empty = today) to list a day's notes, or `frontmatter_read` with a `path` to inspect one file's
   frontmatter — structural lookup, no semantic matching.

Then `read` the relevant hits by `path` (optionally `start_line`/`end_line`; prefer `digest/` paths
for durable knowledge) to pull the content behind a hit. Cite the workspace-relative paths you used.
If nothing useful comes back, say so plainly rather than guessing.

## Server status

To check ReMe is up: call `version` and `health_check`, then summarize the version and the health
snapshot (components, workspace). If the `mcp__reme__…` tools are not available at all, the server
is not running — tell the user to start it with the command above. The plugin connects at
`http://127.0.0.1:2333/mcp`; a different host/port must match the `url` in `plugins/reme/.mcp.json`.

## Proactive & dream job output

`auto_dream`, `proactive`, and `auto_memory` consolidation jobs run server-side and produce
structured results that reach the MCP layer via the standard `response.answer` (human-facing text)
plus `response.metadata` (machine-parseable data) pattern.

- **`response.answer` is always a plain string** suitable for display directly in a reply. For
  `proactive` specifically the format is one or more of, joined with blank lines:
  - `Summary: <one-line digest of what the server observed>` (omitted when identical to Topics)
  - `Topics:` followed by a bulleted list of topic entries when any were extracted
  - `Content:` followed by the raw proactive content string when requested
- **`response.metadata` holds the structured detail**: `topics[]` list with title/reason/evidence/
  keywords/paths fields, `summary`, `skipped`, `content`, plus model-dump fields from
  `ProactiveResult`. When a step needs to branch on *what* topics were found (instead of simply
  showing a summary to the user), read from `metadata.topics` rather than parsing the text answer.

This matches the step fixes in issue #379: LLM-facing answers are self-contained readable strings;
structured consumers use metadata.

## Search answers are self-contained

`search` (and `search_v2`) return readable text even when nothing matches:

- `NO_RESULTS_MESSAGE` when the fused candidate set is empty
- `ALL_RETURNED_MESSAGE` when the dedup layer reports that every previously-seen result has
  already been surfaced

So never treat `answer == ""` as "search ran successfully but produced 0 hits" — instead, look at
the length (or content) of `response.answer`, and **always** cross-check against
`response.metadata.results` (the list of result objects, empty or not) for programmatic use.

## Workspace model

```
daily/    lightly-processed memory: daily facts, conversation summaries
digest/   long-term consolidated knowledge (what recall mainly surfaces)
resource/ external raw materials
```

Consolidation of `daily/` into `digest/` and proactive interest extraction run **server-side** in
the ReMe process (background watchers + dream cron). The plugin does not drive them.
