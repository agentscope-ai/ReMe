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
| TypeScript agent | `@agentscope-ai/reme` |
| Claude Code | MCP + Skill + Stop Hook |
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

The default HTTP service exposes streamable HTTP MCP at `http://127.0.0.1:2333/mcp`. Common tools include `search`, `read`, `traverse`, `list`, `auto_memory`, and `proactive`.

Use `service.jobs` to expose a read-only subset or keep write tools in a separate configuration.

## CLI and Skill

`skills/reme_memory/SKILL.md` defines a general workflow for agents that can run local commands: installation checks, service discovery, retrieval, reading, and persistence boundaries.

It deliberately avoids silently modifying Python environments, stopping unknown processes on port conflicts, writing recalled tool output back as conversation source, or persisting credentials.

## TypeScript, OpenClaw, and DeepSeek Harness

See [TypeScript Agent Integrations](./integrations/typescript.md) for the shared HTTP client, DeepSeek Harness adapter, OpenClaw adapter, and package validation.

## Claude Code

`integrations/claude_code/` provides streamable HTTP MCP configuration, a `reme-memory` Skill, and a Stop hook that calls `auto_memory_cc`. Follow that directory's README for installation.

## Hermes Agent

`integrations/hermes_agent/` provides a memory provider that recalls context before model calls and asynchronously invokes `auto_memory` after each turn.

## Production guidance

- choose a stable absolute `workspace_dir`;
- reuse a service discovered by `reme find_reme`;
- treat `reme help` as the active Job contract;
- apply timeouts and failure logging to writes;
- do not block the host's core response path when memory is temporarily unavailable;
- use authentication, TLS, and a minimal Job allowlist for remote access.
