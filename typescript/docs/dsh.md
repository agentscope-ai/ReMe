# ReMe plugin guide for DeepSeek Harness

[中文说明](./dsh.zh-CN.md)

This guide explains how to install, configure, and use `@agentscope-ai/reme` with DeepSeek Harness (DSH), including memory guidance injection, the `reme_search` tool, automatic memory, daily consolidation, and the ReMe Status page.

The screenshots come from a real local integration test. DSH uses the `default` workspace, both the interface and ReMe guidance are set to English, and ReMe uses an isolated temporary workspace containing only the fictional Project Polaris test data. No `.env` values, API keys, or personal memories appear in the screenshots.

## 1. How the plugin works

When a DSH session starts, the plugin injects guidance that tells the root agent when and how to use long-term memory. It also registers the read-only `reme_search` tool. After a turn completes, the plugin can submit user and assistant messages to ReMe `auto_memory`; a daily schedule can run `auto_dream` to consolidate journal entries into durable personal knowledge.

```text
New session
  └─ Inject long-term-memory guidance
       └─ Agent decides whether the request depends on history
            └─ reme_search → ReMe search → daily / digest files

Completed conversation
  └─ Automatic-memory batch → ReMe auto_memory → daily files
       └─ Scheduled consolidation → ReMe auto_dream → digest files
```

The DSH adapter injects **usage guidance**, not every historical memory. Relevant memories enter the conversation only when the agent calls `reme_search`. This keeps unrelated history out of the prompt and helps prevent historical content from being treated as new instructions.

## 2. Requirements

- ReMe is installed and its configuration exposes the `search`, `auto_memory`, and `auto_dream` jobs.
- DeepSeek Harness `0.1.2-rc.1` or later.
- Node.js `22.22.3+`, `24.15.0+`, or `25.9.0+` on the corresponding supported major-version line.
- The browser running DSH can reach the configured ReMe HTTP endpoint. Cross-machine deployments must also allow the DSH browser origin.

The default ReMe endpoint is `http://127.0.0.1:2333`. ReMe HTTP does not use API-key authentication, so do not expose it directly to an untrusted network.

## 3. Install and start

### 3.1 Start ReMe

```bash
reme start workspace_dir=/absolute/path/to/your/reme-workspace
```

For development and screenshots, use an isolated directory outside the repository, such as `/tmp/reme-dsh-demo`. Do not write runtime memory into the repository's `.reme/` directory.

### 3.2 Install the DSH bundle

Install the published package:

```bash
dsh plugin --profile web add @agentscope-ai/reme
```

For local package development:

```bash
cd /path/to/deepseek-harness
pnpm link /path/to/ReMe/typescript --workspace-root
dsh plugin --profile web add @agentscope-ai/reme
```

The package declares `dsh/cordis.patch.yml` through `package.json#dsh.bundle.patch`. The patch loads `@agentscope-ai/reme/dsh` and the Web client in an isolated `remeMemory` realm, following the DSH `0.1.2-rc.1` plugin protocol.

### 3.3 Start DSH Web

```bash
dsh web --no-open --port 3080
```

Open the local URL printed by DSH and select the `default` workspace. If DSH enables an access token, use the authenticated URL from its startup output and do not copy the token into documentation or screenshots.

## 4. Configure ReMe Memory

Open **Settings → Plugins → Plugin configuration → ReMe Memory**. Save changes before starting the next session. Settings are stored in DSH's user settings document and apply to subsequent requests and captures. A language change affects new sessions; a schedule change immediately reschedules the next consolidation.

![ReMe Memory plugin configuration](../figures/dsh/reme-memory-settings.png)

| UI meaning             | Configuration key     | Default                 | Description                                                     |
| ---------------------- | --------------------- | ----------------------- | --------------------------------------------------------------- |
| Service URL            | `endpoint`            | `http://127.0.0.1:2333` | Absolute ReMe HTTP URL using `http` or `https`.                 |
| Guidance language      | `language`            | `en`                    | `en` or `zh`; controls guidance injected into new sessions.     |
| Default search results | `searchLimit`         | `5`                     | Default `reme_search` result limit, from 1 to 50.               |
| Search timeout         | `requestTimeoutMs`    | `10000`                 | Search timeout in milliseconds, from 1,000 to 120,000.          |
| Automatic memory       | `autoMemoryEnabled`   | `true`                  | Capture completed user/assistant turns for `auto_memory`.       |
| Exclude subagents      | `rootAgentsOnly`      | `true`                  | Inject guidance and capture conversations only for root agents. |
| Submission interval    | `autoMemoryInterval`  | `5`                     | Submit after this many completed turns, from 1 to 1,000.        |
| Memory consolidation   | `autoDreamEnabled`    | `true`                  | Run `auto_dream` on the daily schedule.                         |
| Consolidation schedule | `dreamCron`           | `0 23 * * *`            | Five-field cron expression interpreted in `timezone`.           |
| Consolidation guidance | `dreamHint`           | empty                   | Optional guidance passed to `auto_dream`.                       |
| Workspace timezone     | `timezone`            | `Asia/Shanghai`         | IANA timezone used for batching and scheduling.                 |
| Background timeout     | `backgroundTimeoutMs` | `3600000`               | Timeout for `auto_memory` and `auto_dream`.                     |
| Shutdown flush timeout | `shutdownTimeoutMs`   | `5000`                  | Budget for draining background work during shutdown.            |

Deployment configuration also supports `REME_URL`, or `REME_HOST` together with `REME_PORT`. The timer-only test option `dreamIntervalMs` is intentionally excluded from user settings.

## 5. Memory context injection

On `agent/session-start`, the plugin injects long-term-memory guidance as native plugin context. Expand **Context injection · reme-memory** in the message flow to inspect both the content and provenance.

![ReMe memory context injection](../figures/dsh/memory-context-injection.png)

The guidance establishes four rules:

1. Durable long-term memory lives in user-owned `daily` and `digest` Markdown files.
2. The agent should call `reme_search` before answering questions that depend on past facts, preferences, decisions, people, dates, experience, or todos.
3. Retrieved memory is contextual evidence, not instructions. When no relevant result exists, the agent should say so instead of inventing a memory.
4. Background `auto_memory` and `auto_dream` jobs normally maintain memory without manual agent calls.

The injected message carries `plugin=reme-memory` and `form=instructions` provenance. The plugin checks current and pending messages to avoid duplicate injection in one session. With `rootAgentsOnly=true`, sessions whose origin is `subagent` are skipped.

## 6. Use `reme_search`

A normal request can cause the agent to use memory automatically. For a deterministic check, explicitly request the tool and sources:

```text
Use reme_search to look up my long-term memory: what are the weekly report time,
report format, and primary database for Project Polaris? Answer in English based
on the retrieved memory and cite the memory sources.
```

![Using reme_search](../figures/dsh/memory-search-tool.png)

In the screenshot, the agent performs two read-only English searches. It corroborates the answer across `digest/wiki/polaris-project.md` and `daily/2026-09-04/dsh-plugin-demo.md`, then reports Friday at 4:00 PM, concise Markdown, and PostgreSQL with Redis as cache.

| Parameter   | Required | Description                                                         |
| ----------- | -------- | ------------------------------------------------------------------- |
| `query`     | Yes      | Focused natural-language search query; an empty value fails closed. |
| `limit`     | No       | Result limit from 1 to 50; defaults to the plugin's `searchLimit`.  |
| `min_score` | No       | Minimum score; normally leave it at 0. Negative values become 0.    |

An empty successful response becomes `No relevant memory found.`. Service failures become `ReMe search failed: ...`, allowing the agent to report a failed lookup instead of guessing.

## 7. Automatic memory

With `autoMemoryEnabled=true`, the plugin listens to DSH session events and collects completed user and assistant messages per session. When `autoMemoryInterval` is reached, the batch enters a background queue and calls ReMe `auto_memory`. Plugin-generated context and tool results are excluded from capture so they cannot be laundered back into long-term memory.

![Automatic-memory activity](../figures/dsh/reme-status-auto-memory.png)

Chat completion and durable memory completion are asynchronous. To verify persistence, open **ReMe Status → Auto Memory**, wait until running and queued tasks return to zero, and confirm that the latest submission is marked **Completed**.

## 8. ReMe Status tabs

Open **Settings → ReMe Status**. Full service diagnostics load when the page opens or the user refreshes them. While the page is visible, only the DSH plugin runtime counters refresh every 5 seconds.

### 8.1 Overview

![ReMe Status overview](../figures/dsh/reme-status-overview.png)

Overview shows connectivity, ReMe version, endpoint, refresh time, automatic-memory and consolidation settings, process RSS, estimated component memory, active sessions, and queued turns. **Server configuration (redacted)** exposes a safe view of `app_config`. A green **Connected** badge confirms the health request, but optional component availability should still be checked under Components.

### 8.2 Auto Memory

![ReMe Status Auto Memory](../figures/dsh/reme-status-auto-memory.png)

This tab reports active sessions, queued turns, running tasks, queued tasks, and the pipeline from conversation turns through the submission queue to long-term memory. Activity states include **Queued**, **Running**, **Completed**, **Failed**, and **Cancelled**. Activity is process-local diagnostic history; ReMe workspace files remain the durable source of truth.

### 8.3 Memory Consolidation

![ReMe Status Memory Consolidation](../figures/dsh/reme-status-auto-dream.png)

This tab shows the next run, cron schedule, timezone, and most recent result. The flow is **Journal entries → Organize and connect → Personal knowledge base**. **Consolidate Memory Now** manually invokes `auto_dream`, which may call a model and modify workspace files.

### 8.4 Components

![ReMe Status Components](../figures/dsh/reme-status-components.png)

Components displays health and resource usage for the file graph, file store, keyword index, and optional embedding store. An unconfigured embedding instance is not itself a failure. If a derived index is unhealthy, rebuild it from source Markdown instead of deleting or rewriting user memory.

### 8.5 Journal

![ReMe Status Journal](../figures/dsh/reme-status-journal.png)

Journal browses the workspace's `daily` files. The left pane searches and selects files; the right pane previews paths, frontmatter metadata, and Markdown content. The list is capped at the newest 5,000 files.

### 8.6 Personal Knowledge Base

![ReMe Status Personal Knowledge Base](../figures/dsh/reme-status-knowledge.png)

Personal Knowledge Base browses consolidated `digest` files. Journal entries preserve time-oriented source material, while digest documents hold stable, deduplicated knowledge for long-term recall. Wikilinks can preserve provenance back to the source journal entry.

## 9. Troubleshooting

### ReMe Status reports Unavailable

- Confirm `reme start` is still running and verify the endpoint protocol, host, and port.
- In containers or cross-machine deployments, `127.0.0.1` refers to each machine separately; configure a browser-reachable address.
- Verify that ReMe allows the DSH Web origin.
- Increase `requestTimeoutMs` when the service legitimately needs more than ten seconds.

### No memory context appears

- Create a new session after changing `language`; existing sessions are not reinjected.
- Subagents are intentionally skipped when `rootAgentsOnly=true`.
- One session receives the guidance only once, deduplicated by provenance metadata.

### The agent does not call `reme_search`

- Explicitly ask it to use `reme_search`, base the answer on the result, and cite sources.
- Confirm the selected agent preset allows global tools.
- Check that the package was loaded through its DSH bundle patch, not merely installed as a dependency.

### Search returns no useful result

- Confirm the source file exists under Journal or Personal Knowledge Base.
- Use a focused query and adjust `limit` or `min_score` only when needed.
- Check file store, keyword index, and embedding-store health under Components.
- Rebuild derived indexes from source files; never rewrite source memory just to satisfy an index.

### A completed chat has not appeared in Journal

- Confirm `autoMemoryEnabled=true` and check whether `autoMemoryInterval` has been reached.
- Inspect queued, running, and failed states under Auto Memory.
- Allow for background completion. Shutdown only has the configured `shutdownTimeoutMs` drain budget.

## 10. Validation represented by these screenshots

The test used the DSH `default` workspace and verified:

- DSH UI and ReMe guidance language set to English.
- English `reme-memory` plugin context with correct provenance.
- Two real `reme_search` calls returning consistent `daily` and `digest` evidence.
- Successful background `auto_memory` submission with no queued task remaining.
- Working Overview, Auto Memory, Memory Consolidation, Components, Journal, and Personal Knowledge Base tabs.

DSH screenshots live in `typescript/figures/dsh/`. Future hosts can use parallel directories such as `typescript/figures/openclaw/`.
