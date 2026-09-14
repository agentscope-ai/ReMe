---
title: Agent Integrations
description: Connect ReMe to agents through the CLI, HTTP, MCP, Skills, and host adapters.
---

# Agent Integrations

ReMe keeps memory in an independent service and a user-owned workspace. Multiple agents can call the same memory system without binding storage to one model or host.

## Choose an interface

| Scenario | Recommended interface |
|---|---|
| Local script or hook | ReMe CLI |
| Application backend | HTTP Client |
| Tool-protocol host | MCP |
| DeepSeek Harness | [`@agentscope-ai/reme-dsh-plugin`](./integrations/dsh.md) profile bundle |
| OpenClaw | [`@agentscope-ai/reme-openclaw-plugin`](./integrations/openclaw.md) |
| Claude Code | [Shared HTTP MCP + Skill + Stop Hook](./integrations/claude-code.md) |
| Hermes Agent | Memory provider adapter |
| Codex or another coding agent | `reme_memory` Skill or MCP |

## General memory loop

1. Before answering, call `search` for relevant memory.
2. Use `read` on high-value results and `traverse` when relationships matter.
3. Retain workspace-relative source paths in the answer.
4. At session end, pass source messages to `auto_memory`.
5. Let background or scheduled workflows consolidate daily notes into digest memory.

An empty search result must remain empty; do not present model inference as recalled history.

## MCP

The default HTTP service exposes streamable HTTP MCP at `http://127.0.0.1:2333/mcp`. Common tools include `search`,
`read`, `traverse`, `list`, `auto_memory`, and `proactive_read`.

Use `service.jobs` to expose a read-only subset or keep write tools in a separate configuration.

## CLI and Skill

`skills/reme_memory/SKILL.md` defines a general workflow for agents that can run local commands: installation checks, service discovery, retrieval, reading, and persistence boundaries.

It deliberately avoids silently modifying Python environments, stopping unknown processes on port conflicts, writing recalled tool output back as conversation source, or persisting credentials.

## DeepSeek Harness

Install the self-contained [DeepSeek Harness plugin](./integrations/dsh.md):

```bash
dsh plugin --profile web add @agentscope-ai/reme-dsh-plugin
```

Release links: [Awesome DSH Plugin](https://awesome-dsh-plugin.com/p/agentscope-ai/ReMe--integrations-dsh/) and
[npm](https://www.npmjs.com/package/@agentscope-ai/reme-dsh-plugin).

It injects long-term-memory usage guidance into new root-agent sessions and exposes the read-only `reme_search` tool;
it does not preload the full memory history into the prompt. Completed user/assistant turns can be submitted to
`auto_memory` in background batches, while a timezone-aware schedule runs `auto_dream` to consolidate daily notes.

DSH settings configure the endpoint, guidance language, search limits, capture interval, root-agent filtering, and
consolidation schedule. The ReMe Status page exposes Overview, Auto Memory, Memory Consolidation, Components, Journal,
and Personal Knowledge Base views. Runtime counters are diagnostic state; workspace Markdown remains the durable source
of truth.

## OpenClaw

Install the independently published [OpenClaw plugin](./integrations/openclaw.md):

```bash
openclaw plugins install clawhub:@agentscope-ai/reme-openclaw-plugin
```

Release links: [ClawHub](https://clawhub.ai/agentscope-ai/plugins/reme-openclaw-plugin) and
[npm](https://www.npmjs.com/package/@agentscope-ai/reme-openclaw-plugin). The plugin provides its own host-specific
ReMe HTTP boundary and release lifecycle.

## Claude Code

The [Claude Code plugin](./integrations/claude-code.md) connects every Claude Code window to one ReMe HTTP process at
`http://127.0.0.1:2333/mcp` by default. The `reme-memory` Skill selects among semantic `search`, topological `traverse`,
and state-oriented `daily_list` / `frontmatter_read`, then reads and cites the relevant workspace paths.

On Stop, the hook passes only the Claude Code `session_id` to the server-side `auto_memory_cc` job. On POSIX systems it
detaches the potentially long model call so Claude Code can stop immediately; unreachable-service and other best-effort
failures are written to the plugin log instead of blocking the host. ReMe resolves the local transcript, and repeated
Stop events with no new messages do not create duplicate memory.

## Hermes Agent

`integrations/hermes_agent/` provides a memory provider with HTTP and embedded modes. It recalls context before model calls and asynchronously invokes `auto_memory` after each turn. Its `config_schema.py` is rendered by Hermes' generic memory settings UI.

## Production guidance

- choose a stable absolute `workspace_dir`;
- reuse a service discovered by `reme find_reme`;
- treat `reme help` as the active Job contract;
- apply timeouts and failure logging to writes;
- do not block the host's core response path when memory is temporarily unavailable;
- use authentication, TLS, and a minimal Job allowlist for remote access.
