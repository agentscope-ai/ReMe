---
name: reme-service
description: Use this skill whenever the user references their personal vault (markdown notes managed by the `reme` MCP, service-tier surface), or when there's a meaningful session outcome to record / a question that prior work might answer / a need to clean up the vault. Triggers include "what do I know about X", "did I work on Y before", "save this", "记下", "落盘", "提炼", "vault", any mention of topics/events/methodology, or recognizing that a non-trivial session outcome should be recorded. Skill follows a 4-phase paradigm (Recall / Log / Distill / Maintain) projected onto three MCP tools — `retrieve`, `remember(mode=log|distill)`, `maintain`. The R-M-W loop for Distill and the sweep for Maintain run **inside reme2** (the Ingestor's internal ReActAgent and the Maintainer class respectively).
---

# Vault — service tier

The vault is a personal markdown knowledge base managed by the `reme` MCP server. **Service tier**: three high-level MCP tools, one per memory service. The R-M-W loops for Distill and Maintain run **inside reme2** — you hand off material with the right intent and the Ingestor / Maintainer do the work.

## Business objects

- **Event** — fact log of one session. A folder `events/{YYYY-MM-DD}/{name}/` containing the index `{name}.md` (Memory schema, `lifecycle: streaming`) plus arbitrary **materials** (raw conversation, tool outputs, data dumps). `status: active → distilled → archived`.
- **Topic** — long-lived cognition. Path `topics/{folder}/{name}.md`. Folder topic = `topics/X/X.md` is the cluster's index head.

Lifecycle: **session → event folder (fact + materials) → distill → topic (cognition)**.

## 4-Phase paradigm

Every interaction with the vault falls into one of four phases. The schema, state machine, and wikilink-uniqueness rules are enforced by reme2 underneath — you don't manage them.

### Phase 1: Recall

**What** — retrieve relevant context (chunks ranked by combined relevance + graph proximity).
**Triggers** — intent-driven only: "what do I know about X" / "did I work on Y" / "what's connected to [[Z]]" / task needs prior methodology.
**How (this tier)** — `retrieve(query, max_results?, graph_depth?, seeds?, ...)`. Anchor mode: include `[[Target]]` in the query to seed BFS at that file. Topic-rooted mode: pass `seeds` explicitly.
**Where the work runs** — `reme2` thin primitive (memory_graph_search backend). No LLM.

```
retrieve query="apple valuation" max_results=5
retrieve query="see [[Apple]]" graph_depth=1                    # anchored
retrieve query="any methodology" seeds=["topics/methods/dcf.md"] # topic-rooted
```

Each result chunk carries `graph_hop` so you know how far it sits from your seeds.

### Phase 2: Log

**What** — record event facts + raw materials. Idempotent upsert: same `name` continues the same event folder.
**Triggers** — (a) intent-driven: a meaningful fact / output / decision just landed during the task; (b) **PreCompact hook**: prompt fires asking you to dump volatile state into `materials` before context truncation.
**How (this tier)** — `remember(mode=log, name=..., content?, materials?, topics?, tags?, on_date?)`.
**Where the work runs** — `reme2` thin primitive (sync upsert). No LLM.

**Continuity rule**: pick a stable `name` per logical thread and reuse it across calls. Each call extends the same folder.

```
# First call in the thread
remember
  mode: log
  name: <kebab-case, stable across this thread>
  description: <one line>            # set on first call only
  content: |
    ## ops
    - did X with tool Y
    ## findings
    - Z turned out to be ...
  topics: ["[[X]]", "[[Y]]"]
  tags: [...]
  materials:
    - filename: raw-prompt.md
      content: <user's original prompt>
    - filename: tool-output.txt
      content: <verbose output worth preserving raw>

# Later in the SAME thread (or PreCompact firing)
remember
  mode: log
  name: <same name>
  content: |
    ## follow-up
    - second-pass facts
  materials:
    - filename: tool-output.txt      # collision → auto-suffixed to tool-output-2.txt
      content: <new output>
```

Each subsequent call appends `## Update — {iso}` to the index body, lands new materials as siblings, unions topics/tags. Refuses if the event has already been distilled — pick a fresh `name` from `suggested_name`.

### Phase 3: Distill

**What** — promote active events into the topic graph: read events + materials + linked topics, decide which existing topics to update vs. which deserve a brand-new topic, flip distilled events' status.
**Triggers** — (a) intent-driven: task wraps and the working set is ready; (b) **SessionEnd hook**: prompt fires asking you to call `remember(mode=distill)` once; (c) **Stop hook**: stderr warning if any active events remain undistilled.
**How (this tier)** — `remember(mode=distill, content, hint?, target_path?, metadata?, related_paths?)`. Default `mode` is `distill`, so the parameter can be omitted.
**Where the work runs** — **inside reme2**. The Ingestor's internal ReActAgent reads the working set + linked topics, applies the R-M-W decision rules (SKIP / CONTRADICT / EXTEND / CREATE / STATUS-FLIP), enforces claim-role confidence + wikilink uniqueness + path templates underneath, returns an audit.

**Cold-path rule**: handoff once at task wrap, not per turn. Hand off the working set in two interchangeable forms — `content` (inline material — hint, summary, or raw text) and/or `related_paths` (event folder indexes; the Ingestor follows `## Materials` to read each artifact).

```
remember
  mode: distill                        # default — can be omitted
  content: |
    Distill these active events into the topic graph. Update existing
    topics where extended; create new topics only for genuinely new
    cognitive nodes. Flip each distilled event's status.

    Session summary: <2-3 line recap>
  hint: "End-of-task distillation — minimal edits."
  related_paths:
    - <abs path of event folder 1's index .md>
    - <abs path of event folder 2's index .md>
```

The Ingestor enforces the schema underneath: claim-role topics need `confidence` ∈ {⏳ ✅ ❌}, wikilink uniqueness, path templates. You don't drive any of that — describe intent in `content` / `hint`.

### Phase 4: Maintain

**What** — vault hygiene sweep. Lint (broken wikilinks, schema violations, stem collisions) + decay (move stale events to archive).
**Triggers** — periodic / user suspects vault drift. No hook auto-fires.
**How (this tier)** — `maintain(target_prefix?, ops?, dry_run?, decay_days?)`. Default `dry_run=true` and `ops=["lint","decay"]`.
**Where the work runs** — **inside reme2**. The Maintainer class scans, proposes ops, resolves conflicts, applies (only when `dry_run=false`), returns an audit.

```
maintain                                                 # full vault, lint + decay, dry_run=true
maintain target_prefix="events/2026-04" dry_run=false    # apply within a path subset
maintain ops=["lint"]                                    # diagnostics only
```

Always read the dry-run audit first; re-run with `dry_run=false` if the proposals look right.

## Trigger → Phase quick reference

| Trigger | Phase | What you do |
|---|---|---|
| User asks about prior work / [[X]] | Recall | `retrieve` |
| Fact lands during task | Log | `remember(mode=log, name=...)` |
| **PreCompact hook** fires | Log (urgent dump) | `remember(mode=log, name=..., materials=[...])` + compression guide |
| Task wraps | Distill | `remember(mode=distill, related_paths=[...])` |
| **SessionEnd hook** fires | Log + Distill | final `remember(mode=log)` then `remember(mode=distill)` |
| **Stop hook** warns active events | Distill (compliance) | `remember(mode=distill)` for the listed events |
| User suspects vault drift / periodic | Maintain | `maintain(dry_run=true)` then `dry_run=false` |

## Protocol (the rules reme2 enforces underneath)

Don't manage these — but knowing they exist explains why a write might be refused or a `suggested_name` returned.

@../../protocol.md

## Anti-patterns

- ❌ Picking a fresh `name` on every `remember(mode=log)` call within the same logical thread → fragments the thread. **Reuse the same name.**
- ❌ Reusing the same `name` across genuinely unrelated threads on the same day → silent merge.
- ❌ Calling `remember(mode=distill)` per turn → it's a handoff tool, not a per-turn tool. Once at end-of-task is the rule.
- ❌ Skipping `dry_run=true` on `maintain` for a fresh vault — read what it would do first.
- ❌ Trying to drive schema decisions from your side (claim confidence, path templates, status state machine) — the Ingestor enforces them; pass intent via `content` / `hint` and let it ask if it needs more.
- ❌ Trying to do R-M-W manually with raw `memory_*` tools — those aren't exposed in this tier. Switch to **expert** mode if you need direct primitives.

## What you DON'T have to think about

- The schema (4 axes: lifecycle / scope / source / role) — `remember(mode=distill)` enforces it.
- Wikilink uniqueness — the create gate refuses ambiguity; the Ingestor handles `suggested_name` retries.
- Path templates (`topics/{X}/{X}.md`, `events/{date}/{name}/...`) — the Ingestor / `remember(mode=log)` apply them.
- Status state machine — `remember(mode=distill)` flips it; `maintain` decays.
- Claim-role confidence — the Ingestor refuses claim-role topics without it; pass intent and let it ask.

If you need any of those decisions surfaced (e.g. "I want to manually curate this topic"), switch to **expert** mode which gives you the raw `memory_*` primitives.
