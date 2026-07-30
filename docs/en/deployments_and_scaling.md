# Deployments and Horizontal Scaling

This page collects ReMe's current deployment posture and the project's official
position on horizontal scaling and multi-replica production use. Treat it as a
living design note: the technical boundaries described here are stable, but
specific recommended topologies may change as community backends land.

## Design Principles — Non-Negotiable

ReMe is guided by **"Memory as File, File as Memory."** Whatever deployment
model you choose, two invariants must hold:

1. **User-owned Markdown files remain the source of truth.** Generated state
   (indexes, embeddings, caches, materialized views, external-vector-DB rows)
   is **derived** and must be **fully rebuildable** from the files on disk. If
   you wipe every generated artifact and re-run indexing, you get back an
   equivalent memory system.
2. **The file surface is still readable, diffable, and mergeable by humans.**
   A deployment that can only be inspected through an API is not ReMe — users
   must always be able to `grep`, `git diff`, or edit notes in their editor.

Anything added in the name of productionization must preserve both invariants.

## Current Default Deployment

The default and best-tested topology is a **single long-lived process on one
machine**, with the ReMe workspace directory on fast local storage:

```text
┌────────────────────────────────────────────────────────────┐
│   one machine                                              │
│                                                            │
│   Agent client(s)                                          │
│   ├── Claude Code plugin  ─────┐                          │
│   ├── Codex plugin         ─────┤  localhost 127.0.0.1     │
│   ├── OpenClaw plugin      ─────┤  streamable-http / HTTP  │
│   ├── QwenPaw plugin       ─────┤  service.port=2333       │
│   └── Hermes Agent client  ─────┘                          │
│                         │                                  │
│                         ▼                                  │
│              ┌──────────────────────┐                      │
│              │   ReMe service PID   │                      │
│              │  ├─ watcher loop     │                      │
│              │  ├─ index_updater    │                      │
│              │  ├─ search/HTTP/MCP  │                      │
│              │  ├─ dream cron       │                      │
│              │  └─ auto_memory jobs │                      │
│              └──────────┬───────────┘                      │
│                         │ reads + writes                   │
│                         ▼                                  │
│              ┌──────────────────────┐                      │
│              │  workspace dir (SSD) │                      │
│              │  ├─ daily/           │  source of truth     │
│              │  ├─ digest/          │  ^^^^^^^^^^^^^^^^    │
│              │  ├─ resource/        │                      │
│              │  └─ metadata/        │  rebuildable derived │
│              └──────────────────────┘                      │
└────────────────────────────────────────────────────────────┘
```

Start it with:

```bash
reme start \
  workspace_dir="$HOME/.reme" \
  service.backend=mcp \
  service.transport=streamable-http \
  service.host=127.0.0.1 \
  service.port=2333
```

This model is intentionally simple. All plugin authors (Claude Code, Codex,
OpenClaw, QwenPaw, Hermes Agent) can treat it as the assumed baseline and add
only thin transport wrappers on the client side.

## Official Position on Horizontal Scaling

Based on maintainer feedback in issue
[#386](https://github.com/agentscope-ai/ReMe/issues/386):

> We recognize that authentication, background-job coordination, and shared
> indexes are valid requirements for multi-replica production deployments.
> However, **given our limited maintainer capacity, official horizontal-scaling
> and external-vector-database support are not currently on our roadmap**. We
> will continue to prioritize the core file-native memory experience.
>
> The component architecture is extensible, and we warmly welcome community
> contributions for vector-database or other external backends. To stay aligned
> with ReMe's design principles, such backends must **keep user-owned files as
> the source of truth** and treat external indexes as rebuildable derived
> state. For a substantial implementation, we recommend discussing the design
> with maintainers before opening a PR.

In short: **not on the maintainers' roadmap today, but explicitly welcome as
community-maintained components**, so long as they preserve the two invariants
above. This page exists so that implementers have a clear design target instead
of repeating the same questions in new issues.

## If You Want Multi-User Isolation *Before* Multi-Replica

Before reaching for a multi-service deployment, consider the isolation
primitives already available in a single ReMe process. They cover the majority
of "several users on one box" cases with a small fraction of the operational
cost.

### Option A — one workspace per user (process-level isolation)

Run `N` independent `reme start` processes, each bound to a distinct port and
workspace directory, and route by user in your reverse proxy or client wrapper.

```bash
# user alice
reme start workspace_dir="$HOME/.reme-alice" service.port=2333
# user bob
reme start workspace_dir="$HOME/.reme-bob"   service.port=2334
```

Pros: files never touch; users can carry their own git histories.
Cons: `N` sets of background jobs and `N` watch loops (but indexing is per-user
work anyway so the per-process overhead is modest).

### Option B — one shared workspace, `tags` scoping (lightweight)

Keep a single ReMe process and a single shared workspace, then tag every file
with stable frontmatter tags at write-time and filter on `tags` at recall-time
using the containment-style AND semantics documented in
[Memory Search -> Search Filters](./memory_search.md#search-filters-search_filter).

Write side:

```markdown
---
title: Alice's retro notes
tags:
  - user:alice
  - conv:project-retro
---
```

Recall side:

```json
{
  "name": "search",
  "arguments": {
    "query": "decisions about indexing priority",
    "limit": 5,
    "search_filter": {
      "tags": ["user:alice", "conv:project-retro"]
    }
  }
}
```

Files without `tags` frontmatter remain globally visible (useful for shared
`digest/` knowledge). Per-user or per-conversation files must be written with
their tags — this is best enforced in the wrapper that calls `auto_memory`
instead of relying on manual authoring.

Pros: one process, one set of background jobs; shared knowledge is a natural
side effect.
Cons: no OS-level file isolation; a writer that omits tags leaks a file's
recall surface.

### Option C — hybrid (shared digest, private daily/)

Combine A and B: run one shared workspace that holds public `digest/` and
`resource/`, then tag personal daily notes with `user:<id>` and rely on the
`tags` filter. If stronger isolation is later required, the private daily
notes can move into per-user Option A workspaces without rewriting the shared
digest.

## Community Backend Contributions

If you want to contribute a scalable backend (vector DB pluggable file_store,
a Redis-backed job coordinator, a shared lock service, …), please keep the
following in mind before submitting a PR:

- **Draw a sharp boundary between "what we keep in files" and "what we keep
  externally."**  Markdown/JSONL under the workspace dir is still 100% of the
  durable story; anything external is ephemeral and tagged as such.
- **Provide an explicit "rebuild everything external from files" CLI** and
  document how long it takes on a representative corpus. Without this the
  external state is not "derived" — it's just a new silo.
- **Keep the single-process default path unchanged.** Users who don't need
  scaling shouldn't pay for it in code paths or additional dependencies.
- **Keep config declarative.** New infrastructure choices (remote hosts,
  credentials, timeouts) should live in Hydra config sections under
  `reme/config/` and be overridden via env vars or command-line args, not
  hardcoded.
- **Talk to us first for anything > ~1k LOC.** Open a design issue or comment
  on [#386](https://github.com/agentscope-ai/ReMe/issues/386) with a short
  sketch before writing the implementation. The maintainers can steer you away
  from dead ends and keep the overall architecture consistent.

## Failure Modes to Avoid

When designing a scalable deployment on your own — even one built on the above
options — watch for these classic ReMe-specific footguns:

| Footgun | Why it breaks the invariant |
|---------|-----------------------------|
| Writing facts only to an external index and never to Markdown | Files lose the "source of truth" property; wipe the index and memory is gone. |
| Running `auto_memory` jobs twice for the same session on two replicas | Duplicate daily-note entries; both writes are correct, so humans have to deduplicate manually. Use session-id idempotency or a single writer per conversation. |
| Shared file storage (NFS / SMB) with multiple writers doing in-place edits | ReMe's file watcher and `st_mtime`-based change detection assume a local filesystem. Network filesystems can deliver stale mtimes or lose inotify events and silently miss updates. |
| Tearing down the ReMe process while `auto_memory` or `dream` jobs are in flight | The Stop hooks in each plugin already daemonize and ignore SIGTERM to mitigate this, but aggressive container `SIGKILL` timeouts can still truncate writes. Give the process a long drain interval (≥30s) before shutdown. |
| Per-user embeddings in the same `embedding_store` without namespace tags | Similar to the `tags` footgun: recall returns results the caller shouldn't see. Scope by user tag in the wrapper or use per-user Option A workspaces. |
