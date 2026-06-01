---
name: reme-service
description: Use this skill whenever the user references their personal vault (markdown notes managed by the `reme` MCP, service-tier surface, backed by reme4), or when there's a meaningful session outcome to record / a question that prior work might answer. Triggers include "what do I know about X", "did I work on Y before", "save this", "记下", "落盘", "提炼", "vault", any mention of resource/ / daily/ / digest/ files, or recognizing that a non-trivial session outcome should be recorded. Skill follows a 3-phase paradigm (Recall / Log / Distill) over a 4-tier lifecycle (external channel → resource → daily → digest). Log + Distill route to two SERVICE LAYER MCP tools — `synchronizer` and `digester` — whose internal ReActAgents run the LLM loop INSIDE reme4. Inbound assets from external channels land via `upload` into `resource/<date>/` (service-only). All other tools (search / traverse / file_list / file_read / file_write / file_append / file_stat / frontmatter_*) are shared atomic primitives — whole-file CRUD covers all body changes; `frontmatter` is the one sliced RUD surface (YAML is structured data — surgical key edits cannot be safely emulated with string-substitution on the body).
---

# vault — service tier (reme4)

The vault is a personal markdown knowledge base managed by the `reme` MCP server. **Service tier**: two service-layer LLM-driven tools (`synchronizer`, `digester`) + the service-only resource-ingest primitive (`upload`) + the full shared atomic primitive surface. Log + Distill phases hand off to the service layer; the R-M-W loops run **inside reme4** in those tools' internal ReActAgents.

## Business objects

- **Resource bucket** — passive ingest from external channels. `resource/<YYYY-MM-DD>/` is a flat folder keyed by the day the asset was received, containing the assets themselves (any file type), a `meta.json` array of provenance rows (channel / source / received_at / description), and a derived `<date>.md` view assembled from meta.json. **One ingest path only**: the `upload` tool. Read-only for everything else (synchronizer / digester / hand-edits never write here).
- **Daily note** — hot, streaming fact log of one thread. Single file `daily/<YYYY-MM-DD>/<slug>.md`; everything worth keeping (verbatim user prompt, key tool output, intermediate data) inlined inside the body. One upstream writer per note; every other consumer treats it as read-only. Inbound channel assets do NOT live here — those go to the resource bucket and are referenced via `[[resource/<date>/<name>]]` wikilinks in the note's `## References` section when the task consumes them.
- **digest node** — cold, curated long-lived cognition. `digest/<slug>/<slug>.md` (or nested at any depth: `digest/<scope>/<slug>/<slug>.md`). Each scope folder must contain `<folder>/<folder>.md` as its canonical entry. **Slugs are globally unique under `digest/`** — a folder name appears at most once anywhere in the tree.

Lifecycle: **external channel → `upload` → resource/<date>/ → session work + daily folder → distill → digest node**. Each tier is one-way downstream. Inbound assets are passive (someone sends you a file); daily materials are active (you fetched / produced them during a task); digest entries are distilled cognition. The distill marker is the daily's `status` frontmatter — a **daily-tier convention owned by the Digester** (reme core reserves only `name` / `description`; `status` is an extra used exclusively by Sync/Digester). Convention: absent ≡ `pending`; the Digester flips it to `completed` (or `skipped`) once it has processed the daily. Find unprocessed work by listing `daily/` and `frontmatter_read`-ing each summary — the ones with no `status` are pending.

## Tool surface (service tier)

| Group | Tools | Where the work runs |
|---|---|---|
| **Service layer (LLM-driven)** | `synchronizer`, `digester` | **Inside reme4** — internal ReActAgent |
| **Service-only ingest** | `upload` (external channel → `resource/<date>/`) | reme4 thin primitive (no LLM) |
| Shared retrieve | `search`, `traverse` | reme4 thin primitive (no LLM) |
| Shared read | `file_list`, `file_read`, `file_stat`, `frontmatter_read` | reme4 thin primitive (no LLM) |
| Shared write | `file_write`, `file_append`, `file_edit`, `frontmatter_update`, `frontmatter_delete` | reme4 thin primitive (no LLM) |
| Shared file ops | `file_move`, `file_delete`, `file_download` | reme4 thin primitive (no LLM) |
| Shared daily | `daily_read`, `daily_write`, `daily_list`, `daily_reindex` | reme4 thin primitive (no LLM) |

The shared block is identical to expert tier; what makes this **service** tier is the two service-layer tools at the top plus the service-only `upload` ingest primitive.

## 3-Phase paradigm

### Phase 1: Recall

**What** — retrieve relevant context (chunks ranked by RRF-fused vector + BM25 score, with optional wikilink expansion via the file graph).
**Triggers** — intent-driven only: "what do I know about X" / "did I work on Y" / "what's connected to [[Z]]" / task needs prior methodology.
**How** —
- `search(query, limit?, expand_links?, ...)` for hybrid chunk retrieval.
- `traverse(path, depth?, direction?)` to chase a seed file's wikilink neighborhood.
- `file_list` / `file_read` / `file_stat` / `frontmatter_read` for primary-key reads.

```
search query="auth refactor decisions" limit=5
search query="see [[张三.md]]"
traverse path="digest/zhang-san/zhang-san.md" depth=1
```

### Phase 2: Log (service layer)

**What** — digest the recent conversation slice into a daily note. The Synchronizer's internal ReActAgent picks a slug, writes the note (everything inlined into a single file), and handles continuation (same slug → same file).
**Triggers** — (a) intent-driven: meaningful fact / output / decision just landed; (b) **PreCompact hook**: prompt fires to dump volatile state.
**How** — `synchronizer(messages, note?)`.

```
synchronizer
  messages:
    - {role: user, content: "let's design the auth refactor"}
    - {role: assistant, content: "two options: JWT vs session..."}
    - {role: user, content: "go with JWT + refresh token rotation"}
  note: "auth refactor"               # optional hint to bias the slug
```

The Synchronizer reads the conversation, picks a stable slug (or reuses an existing one when `note` matches), writes `daily/<today>/<slug>.md`, and returns a `SynchronizerResult` audit (`note` path, `summary` of the just-written note, `actions`). Surface the summary verbatim if the user wants to see what landed.

**Surgical edits** (without going through Synchronizer's LLM loop):
- `daily_read(slug, date?)` to probe / merge — returns body in `answer` and the parsed frontmatter dict in metadata. `exists: false` = fresh, `exists: true` = upsert.
- `daily_write(slug, body, frontmatter?, date?, overwrite?)` for a full-note write — `overwrite=false` (default, idempotent skip-if-exists; mirrors the old `daily_resolve` probe) for fresh threads, `overwrite=true` for UPDATE after a `daily_read`. Auto-mkdirs the day folder and refreshes the day index.
- `file_append(path, content)` for cheap end-of-file extensions to trailing sections (`## Progress`, `## Findings`, `## Decisions`) — saves the read-modify-write round-trip.
- `frontmatter_update(path, metadata={key: value, ...})` to merge one or more frontmatter keys (call `daily_reindex` afterward if you touched `name` / `description`).
- `frontmatter_delete(path, keys=[...])` to drop frontmatter keys.
- For mid-body edits on a daily note, `daily_read` then `daily_write overwrite=true`. For non-daily paths, `file_read` then `file_edit` (string substitution) or `file_write` (full body) — there's no body/section slice tool; YAML is the only structured surface that earns its own RUD package.

Use these when you know exactly what to write; use `synchronizer` when you want the service layer to decide what's worth keeping from the conversation.

### Phase 3: Distill (service layer)

**What** — promote daily notes into the digest knowledge graph. The Digester's internal ReActAgent reads each daily note (single-file inline content), looks up existing digest nodes (globally unique slugs — same slug at any nesting depth is the same node), applies the R-M-W decision rules (CREATE / UPDATE / MOVE; mere mentions with no own-node substance are left as-is). Relations are recorded as typed wikilinks in the source node's body (`predicate:: [[X.md]]`); target bodies are never edited (backlinks come from `traverse direction=backward` at query time). After each daily note is processed, the Digester flips its `status` frontmatter to `completed` (or `skipped` if nothing was lifted) — **that flip IS the distill marker** (a daily-tier convention the Digester owns; absent ≡ pending). The Digester scans `daily/` and `frontmatter_read`s each note, picking the ones whose `status` is absent.
**Triggers** — (a) intent-driven: task wraps and the working set is ready; (b) **SessionEnd hook**: prompt fires to call `digester` once.
**How** — `digester(daily_paths, hint?)`.

```
digester
  daily_paths:
    - daily/2026-05-17/auth-refactor
    - daily/2026-05-17/perf-bench
  hint: "End-of-task distillation — focus on the auth decisions; perf-bench is a methodology dump."
```

Returns a `DistillResult` (`used_llm`, `skipped`, `daily_read`, `summary`, `error`). Surface the `summary` verbatim.

**Cold-path rule**: handoff once at task wrap, not per turn.

## Inbound channel ingest (outside the 3-phase loop)

When the user hands you an externally-received asset (file from wechat / email / browser / api / ...), land it in the resource bucket before doing anything else:

```
upload
  path: /tmp/report-q1.pdf
  channel: wechat
  source: design-group
  description: Q1 sales report
```

The tool copies the file into `resource/<today>/<basename>`, appends a `ResourceEntry` to that day's `meta.json`, and regenerates `resource/<today>/<today>.md`. Returns `{date, name, path}` — surface `path` so the user knows where the asset landed. If a downstream task consumes the asset, reference it from the daily note's References section with `[[resource/<date>/<name>]]` (the canonical resource path) rather than inlining the file.

Triggers — user phrases like "save this file", "上传这个", "存一下刚收到的", or a channel hook that hands you an inbound payload.

## Trigger → Phase quick reference

| Trigger | Phase | What you do |
|---|---|---|
| User hands you an inbound asset from an external channel | Ingest | `upload(path=..., channel=..., source=?, description=?)` |
| User asks about prior work / [[X]] | Recall | `search` / `traverse` |
| Fact lands during task | Log | `synchronizer(messages=[...])` |
| Surgical edit needed | Log | `daily_read` / `daily_write` / `file_append` / `frontmatter_update` / `frontmatter_delete` / `daily_reindex` |
| **PreCompact hook** fires | Log (urgent dump) | `synchronizer(messages=[...], note=...)` |
| Task wraps | Distill | `digester(daily_paths=[...])` |
| **SessionEnd hook** fires | Log + Distill | `synchronizer` then `digester` |

## Protocol (the rules every write must respect)

@../../../../protocol.md

## Anti-patterns

- ❌ Picking a fresh `note` slug on every `synchronizer` call within the same logical thread → fragments the thread. **Reuse the slug.**
- ❌ Calling `digester` per turn → it's a handoff tool, not a per-turn tool. Once at end-of-task is the rule.
- ❌ Calling `digester` on a daily whose `status` is already `completed` or `skipped` → it'll be a no-op; don't keep re-pushing. (The Digester's own pending scan — `file_list` + per-item `frontmatter_read` — filters those out for you.)
- ❌ Creating a new `digest/X/x.md` when `X` already exists somewhere else under `digest/` (e.g. `digest/people/X/x.md`) — slugs are globally unique; reuse the existing node and fold the new facts in.
- ❌ Manually `file_write`-ing under `digest/` instead of going through `digester` — digest nodes are the Digester's domain. (You can still do it for one-off corrections; just don't bypass the service layer for routine distillation.)
- ❌ Writing `status` from outside the Digester — `status` is a Digester-owned daily-tier convention (enum `pending` / `completed` / `skipped`; absent ≡ pending). Flipping it from a hand-written tool call makes the note look already-processed (the Digester's pending scan skips it) and the next `digester` invocation never picks it up.
- ❌ Using `file_write` (full-file replacement) to flip one frontmatter key — use `frontmatter_update`.
- ❌ Using `file_write` to extend trailing sections like `## Progress` — use `file_append`; saves the R-M-W round-trip and the prompt tokens of echoing the whole body back.
- ❌ Writing under `resource/` from anything other than `upload` — that bucket is the passive ingest contract. Hand-edits / `synchronizer` / `digester` must never touch it.
- ❌ Inlining an inbound asset into the daily note body — leave it in `resource/<date>/<name>` and reference it via `[[resource/<date>/<name>]]` in the note's `## References` section. Daily notes are single-file; inbound assets stay in `resource/`.
- ❌ Writing short-form (`[[Alice]]`) or no-extension (`[[k/x]]`) wikilinks — they don't resolve. Always full path relative to the vault with `.md`: `[[digest/alice/alice.md]]`.

## What you DON'T have to think about

- Slug uniqueness within a thread — `synchronizer` / `digester` pick paths and reuse existing notes; wikilinks are literal full paths, so two different paths never silently merge.
- Status flips — the Digester writes `status=completed` (or `skipped`) per processed daily note; that frontmatter flag IS the distill marker, so it's never optional but you never write it yourself.
- Frontmatter schema — only `name` / `description` / `status` are reserved (all optional); the protocol defines opinionated default axes (`lifecycle` / `scope` / `source` / `role`) but enforcement is caller-side, not protocol-side.
- Pending-vs-digest bookkeeping — the digester scans `daily/` and `frontmatter_read`s each note to find ones whose `status` is absent; it maintains the queue.

If you need fine-grained control over every R-M-W decision visible in the main session's tool log, switch to [reme-expert](../reme-expert) — same shared tools, no service layer, plus a subagent that owns the Distill LLM loop in its own context window.
