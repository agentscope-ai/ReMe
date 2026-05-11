# reme — Claude Code marketplace

Two paradigms for managing a markdown vault from Claude Code, packaged as
two installable plugins. **The 4-phase work paradigm is identical
across both** — only the locus of computation for the LLM-driven phases
(Distill, Maintain) differs.

## The 4-phase paradigm (both plugins)

Every interaction with the vault falls into one of four phases:

| Phase | What | Trigger |
|---|---|---|
| **Recall** | Retrieve relevant context (chunks ranked by relevance + graph proximity). | Intent-driven: user asks about prior work or the task needs prior context. |
| **Log** | Record event facts + raw materials into an idempotent event-folder upsert. | (a) intent-driven during the task; (b) **PreCompact hook** to dump volatile state. |
| **Distill** | Promote events into the topic graph via R-M-W (read → mutate → write). | (a) intent-driven at task wrap; (b) **SessionEnd hook**; (c) **Stop hook** warns if skipped. |
| **Maintain** | Vault hygiene sweep — lint + decay. | Periodic / user suspects drift. No hook auto-fires. |

The triggers + hook set are **identical** in both plugins. What changes
is which tool / mechanism realizes each phase, and where the LLM-driven
loops actually run.

## Service tier vs Expert tier

| Phase | [`reme-service`](./plugins/reme-service) — work runs **inside reme2** | [`reme-expert`](./plugins/reme-expert) — work runs **outside reme2** |
|---|---|---|
| Recall | `retrieve` MCP tool | `memory_search` / `memory_graph_search` MCP tools, `/reme-recall` slash |
| Log | `remember(mode=log, name=...)` MCP tool | `sync(name=...)` MCP tool (+ `memory_update` / `memory_property_update` for surgical edits) |
| Distill | `remember(mode=distill, ...)` MCP tool → **Ingestor's internal ReActAgent** runs the R-M-W loop inside reme2 | `/reme-distill` slash → **`reme-distiller` subagent** runs the R-M-W loop outside reme2 (own context window) |
| Maintain | `maintain(...)` MCP tool → **Maintainer class** runs the sweep inside reme2 | `/reme-clean` slash → **`reme-curator` subagent** runs the sweep outside reme2 |

Both plugins ship the same hook set: **PreCompact / SessionEnd / Stop**.
The prompt content differs only in which tool / slash command to call.
Recall and Log are thin no-LLM primitives in both modes — only the MCP
names differ.

Both plugins are built on [ReMe2](../reme2) (file_store + watcher + the
three memory services) and the 4-axis Memory schema in
`reme2/memory/schema/`. Both ship a copy of the canonical
`reme2/memory/protocol.md` so the rules (4-axis schema, status state
machine, claim-role confidence, wikilink uniqueness, R-M-W decision
tree) are the same single source of truth across plugin and reme2.

## Pick one

| | `reme-service` | `reme-expert` |
|---|---|---|
| Cognitive load | Low — call three high-level tools and trust the loops | High — agent runs its own R-M-W following the protocol |
| Best when | The agent should focus on its primary task; memory is plumbing | The agent should be a first-class citizen of the vault — every R-M-W decision visible and auditable |
| Subagents | None | `reme-distiller`, `reme-curator` |
| Slash commands | None (the three MCP tools are enough) | `/reme-distill`, `/reme-recall`, `/reme-clean` |

## Install

```text
/plugin marketplace add /Users/huangsen/codes/ReMe/reme-plugin
/plugin install reme-service     # OR
/plugin install reme-expert
```

Restart Claude Code so the MCP server / hooks pick up.

Set `VAULT_PATH` so the MCP server points at your vault directory:

```bash
export VAULT_PATH=/path/to/your/vault
```

If unset, falls back to `./vault` relative to launch CWD (the dev default).

## Layout

```
reme-plugin/
├── .claude-plugin/
│   └── marketplace.json         # lists both plugins
├── README.md                    # this file
└── plugins/
    ├── reme-service/            # Service-tier
    │   ├── .claude-plugin/plugin.json
    │   ├── .mcp.json            # config=reme2/config/service.yaml
    │   ├── protocol.md          # canonical protocol (copied from reme2/)
    │   ├── skills/reme-service/SKILL.md
    │   ├── hooks/               # PreCompact / SessionEnd / Stop
    │   └── README.md
    └── reme-expert/             # Expert-tier
        ├── .claude-plugin/plugin.json
        ├── .mcp.json            # config=reme2/config/expert.yaml
        ├── protocol.md          # canonical protocol (copied from reme2/)
        ├── skills/reme-expert/SKILL.md
        ├── hooks/               # PreCompact / SessionEnd / Stop
        ├── agents/              # reme-distiller, reme-curator subagents
        ├── commands/            # /reme-distill, /reme-recall, /reme-clean
        └── README.md
```

Each plugin's MCP server is `python -m reme2.mcp.server` with
`PYTHONPATH=${CLAUDE_PLUGIN_ROOT}/../../..` so it resolves the sibling
`reme2/` package without pip install. The two plugins differ only in
which config they pin (`service.yaml` vs `expert.yaml`).

## Switching modes

Both modes write through the same MFS engine and read the same files,
so the vault is portable across modes. To switch:

1. `/plugin uninstall reme-{service|expert}`
2. `/plugin install reme-{expert|service}`
3. Restart Claude Code.

You can keep both installed simultaneously as long as you don't run two
MCP servers at the same vault — the file_store cache will fight.

Switch from **service → expert** when:
- You need fine-grained control over what gets created/edited.
- You want every memory mutation visible in the main session's tool log.
- You want subagents to handle distillation as bounded background work.

Switch from **expert → service** when:
- The agent's main task is what matters; memory should be invisible plumbing.
- You don't want to maintain or read the protocol.
- You're OK with the Ingestor's defaults around topic creation, claim confidence, and wikilink uniqueness.
