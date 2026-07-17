---
name: reme-memory
description: Use ReMe as file-native long-term memory in Codex. RECALL — search ReMe before answering questions about past conversations, preferences, project history, or decisions.
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

## Workspace model

```
daily/    lightly-processed memory: daily facts, conversation summaries
digest/   long-term consolidated knowledge (what recall mainly surfaces)
resource/ external raw materials
```

Consolidation of `daily/` into `digest/` and proactive interest extraction run **server-side** in
the ReMe process (background watchers + dream cron). The plugin does not drive them.
