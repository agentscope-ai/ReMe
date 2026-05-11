# reme-expert

Expert-tier markdown vault management for Claude Code. Claude Code's
agent runs the R-M-W loop directly using raw `memory_*` MCP primitives,
guided by the `reme-expert` protocol skill. The LLM-driven Distill loop
and the agentic Maintain sweep run **outside reme2** in dedicated
subagents (`reme-distiller`, `reme-curator`) that own their own context
windows.

For the simpler service-tier alternative where reme2's internal Ingestor
+ Maintainer own those loops, install [reme-service](../reme-service) instead.

## Install

```text
/plugin marketplace add /Users/huangsen/codes/ReMe/reme-plugin
/plugin install reme-expert
```

```bash
export VAULT_PATH=/path/to/your/vault
```

## What's in the box

### MCP server (`reme`)

16 tools from `reme2/config/expert.yaml` — every memory_* primitive plus
`sync` for event-folder upsert:

| Group | Tools | Phase |
|---|---|---|
| Hot-write | `sync` | Log |
| Reads | `memory_search`, `memory_graph_search`, `memory_get`, `memory_list`, `memory_links`, `memory_backlinks`, `memory_resolve_wikilink`, `memory_count_tokens`, `memory_lint` | Recall |
| Writes | `memory_create`, `memory_update`, `memory_property_update`, `memory_rename`, `memory_delete`, `memory_archive` | Log (surgical) / Distill (subagent uses) |

`ingest` is deliberately NOT exposed in this profile — Distill runs in
the `reme-distiller` subagent, not in reme2's internal ReActAgent.

### Skill

`reme-expert` (auto-invoked) — the agent's protocol playbook. Describes
the 4-phase paradigm (Recall / Log / Distill / Maintain), the trigger
table, and which tool / slash command realizes each phase. Symmetrical
with `reme-service`'s skill: same chapter structure, same triggers,
just different tools to call.

### Subagents

| Subagent | Phase | Role |
|---|---|---|
| `reme-distiller` | Distill | Reads active events + materials + linked topics, applies R-M-W decision rules (SKIP / CONTRADICT / EXTEND / CREATE / STATUS-FLIP), returns audit. Auto-spawned by SessionEnd hook; also invokable via `/reme-distill`. |
| `reme-curator` | Maintain | Vault hygiene sweep. Lint + decay; default `dry_run=true`. Invokable via `/reme-clean`. |

Both subagents transclude `../protocol.md` so they see the same rules as
the main agent's skill.

### Slash commands

| Command | Purpose |
|---|---|
| `/reme-distill [hint or paths]` | Spawn `reme-distiller` (auto-fires at SessionEnd; manual invocation available). |
| `/reme-recall <query>` | Explicit deep `memory_graph_search` (8 hits, depth 1). |
| `/reme-clean [target_prefix] [dry_run]` | Spawn `reme-curator` for hygiene sweep. |

### Hooks (parallel to reme-service)

| Event | Action |
|---|---|
| **PreCompact** | Prompt: call `sync(materials=[...])` to dump volatile state into the current thread's event folder, reusing the thread's `name` for upsert; then output a compression guide. |
| **SessionEnd** | Prompt: call `sync` to capture remaining facts, then **invoke `/reme-distill`** to spawn the distiller subagent. |
| **Stop** | `active_events_check.py` — warns via stderr if any `status: active` events remain, suggests `/reme-distill`. |

There are **no SessionStart / UserPromptSubmit auto-recall hooks** — the
agent's skill tells it when to call `memory_search` itself, which is
more accurate than blanket injection. Use `/reme-recall` when you need
deeper recall than your inline search would surface.

## Layout

```
plugins/reme-expert/
├── .claude-plugin/plugin.json
├── .mcp.json                          # config=reme2/config/expert.yaml
├── protocol.md                        # canonical protocol; transcluded by SKILL via @../../protocol.md and by subagents via @../protocol.md
├── skills/reme-expert/SKILL.md
├── hooks/
│   ├── hooks.json                     # PreCompact / SessionEnd / Stop
│   └── active_events_check.py
├── agents/
│   ├── reme-distiller.md
│   └── reme-curator.md
├── commands/
│   ├── reme-distill.md
│   ├── reme-recall.md
│   └── reme-clean.md
└── README.md                          # this file
```

`.mcp.json` resolves the sibling `reme2/` package via
`PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/../../..` (marketplace root → repo
root). No pip install needed.

`protocol.md` is a copy of the canonical `reme2/memory/protocol.md`. The
SKILL transcludes it from `skills/reme-expert/` via `@../../protocol.md`;
the subagents transclude it from `agents/` via `@../protocol.md`. This
local copy keeps the references stable after marketplace install.

## Workflow at a glance

```
对话过程            → 模型按需调 memory_search / memory_graph_search；
                     /reme-recall 触发更深召回
任务执行中          → sync 持续 upsert；surgical 时 memory_update / memory_property_update
PreCompact (hook)  → 提示调 sync dump materials + 输出压缩指引
任务完成           → /reme-distill → reme-distiller 子代理在独立 context 中跑 R-M-W
SessionEnd (hook)  → 提示 sync 收尾 + invoke /reme-distill (passing event paths)
Stop (hook)        → active_events_check.py 提醒未 distill 的 active events
periodic           → /reme-clean → reme-curator 子代理跑 lint + decay
```

## When to switch to service mode

Switch to [reme-service](../reme-service) when:

- You don't want to maintain or read the protocol skill — let the
  Ingestor's defaults decide.
- The agent should focus on its primary task and treat memory as plumbing.
- You're OK trusting reme2's auto-decisions on topic creation, claim
  confidence, and wikilink uniqueness.
