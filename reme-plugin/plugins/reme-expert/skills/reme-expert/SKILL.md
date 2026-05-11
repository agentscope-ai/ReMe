---
name: reme-expert
description: Use this skill whenever the user references their personal vault (markdown notes managed by the `reme` MCP, expert-tier surface), or when there's a meaningful session outcome to record / a question that prior work might answer / a need to clean up the vault. Triggers include "what do I know about X", "did I work on Y before", "save this", "记下", "落盘", "提炼", "vault", any mention of topics/events/methodology, or recognizing that a non-trivial session outcome should be recorded. Skill follows a 4-phase paradigm (Recall / Log / Distill / Maintain) projected onto raw `memory_*` MCP primitives + `sync` (event log) + slash commands `/reme-distill` (auto-fired at SessionEnd) / `/reme-recall` / `/reme-clean`. Distill and Maintain run **outside reme2** in dedicated subagents (`reme-distiller`, `reme-curator`) that own their own context windows.
---

# Vault — expert tier

The vault is a personal markdown knowledge base managed by the `reme` MCP server. **Expert tier**: raw `memory_*` primitives + `sync` for event logging, plus slash commands that spawn subagents for the LLM-driven loops. The R-M-W loops for Distill and Maintain run **outside reme2** in subagents you (Claude Code) drive directly via the same primitives.

## Business objects

- **Event** — fact log of one session. A folder `events/{YYYY-MM-DD}/{name}/` containing the index `{name}.md` (Memory schema, `lifecycle: streaming`) plus arbitrary **materials** (raw conversation, tool outputs, data dumps). `status: active → distilled → archived`.
- **Topic** — long-lived cognition. Path `topics/{folder}/{name}.md`. Folder topic = `topics/X/X.md` is the cluster's index head.

Lifecycle: **session → event folder (fact + materials) → distill → topic (cognition)**.

## 4-Phase paradigm

Every interaction with the vault falls into one of four phases. The schema, state machine, and wikilink-uniqueness rules apply to every write you make — see the protocol section below.

### Phase 1: Recall

**What** — retrieve relevant context (chunks ranked by combined relevance + graph proximity).
**Triggers** — intent-driven only: "what do I know about X" / "did I work on Y" / "what's connected to [[Z]]" / task needs prior methodology.
**How (this tier)** — `memory_search(query, ...)` for hybrid vector + keyword; `memory_graph_search(query, seeds?, graph_depth?, ...)` when you want context expansion through wikilinks. For deeper recall (8 hits, depth 1) invoke `/reme-recall <query>`.
**Where the work runs** — `reme2` thin primitive (FileGraph + FTS + vector store). No LLM.

```
memory_search query="apple valuation" max_results=5
memory_graph_search query="see [[Apple]]" graph_depth=1
/reme-recall apple valuation around [[Apple]]
```

Each result chunk carries `graph_hop` so you know how far it sits from your seeds.

Use `memory_get` / `memory_list` / `memory_links` / `memory_backlinks` / `memory_resolve_wikilink` for primary-key reads when you already know the path or want to follow a specific edge.

### Phase 2: Log

**What** — record event facts + raw materials. Idempotent upsert: same `name` continues the same event folder.
**Triggers** — (a) intent-driven: a meaningful fact / output / decision just landed during the task; (b) **PreCompact hook**: prompt fires asking you to dump volatile state into `materials` before context truncation.
**How (this tier)** — `sync(name=..., content?, materials?, topics?, tags?, on_date?, description?)`.
**Where the work runs** — `reme2` thin primitive (sync upsert). No LLM.

**Continuity rule**: pick a stable `name` per logical thread and reuse it across calls. Each call extends the same folder.

```
# First call in the thread
sync
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
sync
  name: <same name>
  content: |
    ## follow-up
    - second-pass facts
  materials:
    - filename: tool-output.txt      # collision → auto-suffixed to tool-output-2.txt
      content: <new output>
```

Each subsequent call appends `## Update — {iso}` to the index body, lands new materials as siblings, unions topics/tags. Refuses if the event has already been distilled — pick a fresh `name` from `suggested_name`.

**Inline surgical edits during the task** (not Distill — Distill is a bigger loop):
- Change one frontmatter field on an existing topic → `memory_property_update path=... key=... value=...`
- Edit one body snippet on an existing topic → `memory_update path=... old_string=... new_string=...`
- These are still Phase 2 (you're recording a small fact); use them when the change is small and obvious. For multi-step R-M-W, defer to Phase 3.

### Phase 3: Distill

**What** — promote active events into the topic graph: read events + materials + linked topics, decide which existing topics to update vs. which deserve a brand-new topic, flip distilled events' status.
**Triggers** — (a) intent-driven: task wraps and the working set is ready; (b) **SessionEnd hook**: prompt fires asking you to invoke `/reme-distill`; (c) **Stop hook**: stderr warning if any active events remain undistilled.
**How (this tier)** — invoke `/reme-distill [hint or event paths]`. The slash command spawns the `reme-distiller` subagent; it reads the protocol, runs the R-M-W loop using raw `memory_*` tools, returns an audit summary.
**Where the work runs** — **outside reme2**. The `reme-distiller` subagent is a Claude Code agent with its own context window — main session doesn't bloat with the 5-10 tool-call R-M-W sequence. The subagent uses the same `memory_*` primitives this skill describes; it just runs them in isolation and reports back.

**Cold-path rule**: handoff once at task wrap, not per turn. The SessionEnd hook auto-fires the prompt — your job there is just to (1) `sync` final state and (2) invoke `/reme-distill`, passing event paths from this session if you have them.

```
/reme-distill end-of-task distillation. event folders this session:
              events/2026-05-09/<name1>/<name1>.md
              events/2026-05-09/<name2>/<name2>.md
```

The subagent owns the decision rules (SKIP / CONTRADICT / EXTEND / CREATE / STATUS-FLIP) and the schema enforcement (claim-role confidence, wikilink uniqueness, path templates). Surface its audit verbatim — it's the source of truth for what changed.

### Phase 4: Maintain

**What** — vault hygiene sweep. Lint (broken wikilinks, schema violations, stem collisions) + decay (move stale events to archive).
**Triggers** — periodic / user suspects vault drift. No hook auto-fires.
**How (this tier)** — invoke `/reme-clean [target_prefix] [dry_run]`. The slash command spawns the `reme-curator` subagent; it runs `memory_lint` for diagnostics + uses `memory_rename` / `memory_archive` / `memory_property_update` to apply fixes. Default `dry_run=true`.
**Where the work runs** — **outside reme2**. `memory_lint` is a thin reme2 projection of the Maintainer's diagnostics, but the *application* of fixes is agentic — the curator decides which renames / archives to actually apply and uses raw write primitives, all in the subagent's own context.

```
/reme-clean                              # whole vault, dry_run=true
/reme-clean events/2026-04 dry_run=false # apply within a path subset
```

Always read the dry-run audit first; re-run with `dry_run=false` if the proposals look right.

For a quick read-only diagnostic without spawning a subagent, you can call `memory_lint` directly — but apply-side fixes still go through `/reme-clean`.

## Trigger → Phase quick reference

| Trigger | Phase | What you do |
|---|---|---|
| User asks about prior work / [[X]] | Recall | `memory_search` / `memory_graph_search` / `/reme-recall` |
| Fact lands during task | Log | `sync(name=...)` |
| Small surgical edit needed | Log | `memory_update` / `memory_property_update` inline |
| **PreCompact hook** fires | Log (urgent dump) | `sync(name=..., materials=[...])` + compression guide |
| Task wraps | Distill | `/reme-distill` |
| **SessionEnd hook** fires | Log + Distill | final `sync` then `/reme-distill <event paths>` |
| **Stop hook** warns active events | Distill (compliance) | `/reme-distill` for the listed events |
| User suspects vault drift / periodic | Maintain | `/reme-clean` (dry_run=true) then `dry_run=false` |

## Protocol (the rules every write must respect)

These apply to every `memory_create` / `memory_update` / `memory_property_update` / `sync` you make and to every R-M-W decision the distiller subagent runs.

@../../protocol.md

## Anti-patterns

- ❌ Picking a fresh `name` on every `sync` call within the same logical thread → fragments. **Reuse the same name.**
- ❌ Reusing the same `sync` `name` across genuinely unrelated threads on the same day → silent merge.
- ❌ Running R-M-W inline in the main session at end-of-task — that's what `/reme-distill` exists for. Use it so the loop runs in the subagent's own context.
- ❌ Using `memory_update` to edit YAML frontmatter — use `memory_property_update`.
- ❌ Calling `memory_create` for a topic whose stem already resolves to another path — wikilink-uniqueness gate refuses; rename or pick a domain-specific qualifier.
- ❌ Writing a thesis/model/questions topic without `confidence` — schema validation refuses; either add it before creating, or let the distiller surface the gap.
- ❌ Writing event files manually with `memory_create` under `events/` — use `sync` so path template + Materials footer + idempotent upsert all run.
- ❌ Forcing event status straight from `active` to `archived` — single-direction state machine; archive only after `distilled`.
- ❌ Skipping `dry_run=true` on `/reme-clean` — read the plan first.

## What's parallel to service mode (for comparison)

| Phase | Service tier (work runs **inside reme2**) | Expert tier (work runs **outside reme2**) |
|---|---|---|
| Recall | `retrieve` | `memory_search` / `memory_graph_search` / `/reme-recall` |
| Log | `remember(mode=log, name=...)` | `sync(name=...)` |
| Distill | `remember(mode=distill, ...)` (Ingestor's internal ReActAgent) | `/reme-distill` (Claude Code subagent) |
| Maintain | `maintain(...)` (Maintainer class) | `/reme-clean` (Claude Code subagent) |

The 4 phases and their triggers are identical across modes; only the locus of computation for Distill and Maintain differs (and Recall/Log have different MCP names).
