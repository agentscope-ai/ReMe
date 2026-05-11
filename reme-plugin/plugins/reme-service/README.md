# reme-service

Service-tier markdown vault management for Claude Code. Three high-level
MCP tools — `retrieve` / `remember` / `maintain` — backed by reme2's
internal Ingestor + Maintainer. The agent calls the tools with intent;
the LLM-driven R-M-W loop (Distill) and the sweep loop (Maintain) run
**inside reme2**.

For the agentic alternative where Claude Code's own subagents run those
loops outside reme2, install [reme-expert](../reme-expert) instead.

## Install

```text
/plugin marketplace add /Users/huangsen/codes/ReMe/reme-plugin
/plugin install reme-service
```

```bash
export VAULT_PATH=/path/to/your/vault
```

## What's in the box

### MCP server (`reme`)

Three tools projected from the three memory services
(`reme2/config/service.yaml`):

| Tool | Phase | Backend |
|---|---|---|
| `retrieve(query, max_results?, graph_depth?, seeds?, ...)` | Recall | thin (memory_graph_search) |
| `remember(mode=log\|distill, name?, content, materials?, related_paths?, ...)` | Log / Distill | sync upsert (mode=log, no LLM) / Ingestor's internal ReActAgent (mode=distill) |
| `maintain(target_prefix?, ops?, dry_run?, decay_days?)` | Maintain | Maintainer class |

### Skill

`reme-service` (auto-invoked) — describes the 4-phase paradigm
(Recall / Log / Distill / Maintain) and how each phase maps to the three
tools. Symmetrical with `reme-expert`'s skill: same chapter structure,
same triggers, just different tools to call.

### Hooks (parallel to reme-expert)

| Event | Action |
|---|---|
| **PreCompact** | Prompt: call `remember(mode=log, materials=[...])` to dump volatile state into the current thread's event folder, reusing the thread's `name` for upsert; then output a compression guide. |
| **SessionEnd** | Prompt: call `remember(mode=log)` to capture remaining facts, then call `remember(mode=distill, related_paths=[...])` once to hand off the working set to the Ingestor. |
| **Stop** | `active_events_check.py` — warns via stderr if any `status: active` events remain, suggests `remember(mode=distill)`. |

There are **no SessionStart / UserPromptSubmit auto-recall hooks** — the
agent's skill tells it when to call `retrieve` itself, which is more
accurate than blanket injection.

## Layout

```
plugins/reme-service/
├── .claude-plugin/plugin.json
├── .mcp.json                       # config=reme2/config/service.yaml
├── protocol.md                     # canonical protocol, transcluded by SKILL via @../../protocol.md
├── skills/reme-service/SKILL.md
├── hooks/
│   ├── hooks.json                  # PreCompact / SessionEnd / Stop
│   └── active_events_check.py
└── README.md                       # this file
```

`.mcp.json` resolves the sibling `reme2/` package via
`PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/../../..` (marketplace root → repo
root). No pip install needed.

`protocol.md` is a copy of the canonical `reme2/memory/protocol.md`. The
same file is read by reme2's internal Ingestor (injected as `{protocol}`
into the ReActAgent's sys_prompt) — this plugin ships its own copy so
the SKILL's `@../../protocol.md` transclusion works after marketplace
install.

## Workflow at a glance

```
对话过程            → 模型按需调 retrieve（intent-driven）
任务执行中          → remember(mode=log) 持续 upsert；materials 装原始素材（确定性、零 LLM）
PreCompact (hook)  → 提示调 remember(mode=log) dump materials + 输出压缩指引
任务完成           → remember(mode=distill) 一次 → Ingestor 内部 ReActAgent 跑 R-M-W → 返回 audit
SessionEnd (hook)  → 提示 remember(mode=log) 收尾 + remember(mode=distill) handoff
Stop (hook)        → active_events_check.py 提醒未 distill 的 active events
periodic           → maintain(dry_run=true) 看一眼，必要时 dry_run=false 应用
```

## When to switch to expert mode

The service plugin hides the schema, claim confidence, wikilink
uniqueness, path templates, and status state machine — `remember(mode=
distill)` enforces them all underneath. Switch to
[reme-expert](../reme-expert) when:

- You need fine-grained control over what gets created/edited.
- You want every memory mutation visible in the main session's tool log
  (vs hidden inside the Ingestor's internal ReActAgent).
- You want subagents to handle distillation as bounded background work
  with their own context windows.
