# ReMe memory for OpenClaw

[中文说明](./openclaw.zh-CN.md)

ReMe gives OpenClaw file-native long-term memory while keeping durable memory in a workspace you own. The plugin uses
OpenClaw's native lifecycle, hooks, tools, and memory slot.

## What the plugin does

- Registers `reme_search` for explicit memory lookup.
- Recalls relevant memory before conversational root-agent runs.
- Captures completed user/assistant turns in background batches.
- Runs optional daily `auto_dream` memory consolidation in the workspace timezone.
- Excludes subagent, cron, heartbeat, memory, and overflow runs by default.
- Wraps recalled content in `<reme-context>` and marks it as untrusted historical data.

## Requirements

- OpenClaw `2026.7.1` or later.
- Node.js `22.22.3+`, `24.15.0+`, or `25.9.0+` on the corresponding supported major-version line.
- A running ReMe HTTP service with the `search`, `auto_memory`, and `auto_dream` jobs.

## Start ReMe

Install ReMe and start its local HTTP service:

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace
```

The default endpoint is `http://127.0.0.1:2333`. ReMe's HTTP service does not use API-key authentication, so keep it
on loopback or another trusted network unless you provide a protected proxy boundary.

## Install the OpenClaw plugin

Install explicitly from ClawHub:

```bash
openclaw plugins install clawhub:@agentscope-ai/reme
```

When another memory plugin is enabled, select `reme` for `plugins.slots.memory`. OpenClaw remains authoritative for
conversation access and prompt-injection permissions; grant them to ReMe when your policy requires explicit consent.
The plugin does not rewrite Gateway configuration.

## Configuration

| Option                | Default                 | Meaning                                       |
| --------------------- | ----------------------- | --------------------------------------------- |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP service URL                         |
| `language`            | `en`                    | Memory guidance language: `en` or `zh`        |
| `autoRecall`          | `true`                  | Recall before conversational root-agent runs  |
| `searchLimit`         | `5`                     | Maximum search results                        |
| `recallMinScore`      | `0`                     | Minimum automatic-recall score                |
| `autoMemoryEnabled`   | `true`                  | Capture completed conversational turns        |
| `autoMemoryInterval`  | `5`                     | Submit after this many completed turns        |
| `autoDreamEnabled`    | `true`                  | Enable daily memory consolidation             |
| `dreamCron`           | `0 23 * * *`            | Daily schedule in the workspace timezone      |
| `dreamHint`           | empty                   | Optional guidance sent to `auto_dream`        |
| `rootAgentsOnly`      | `true`                  | Exclude subagents from guidance and capture   |
| `timezone`            | `Asia/Shanghai`         | IANA timezone used for batches and scheduling |
| `requestTimeoutMs`    | `10000`                 | Recall and explicit-search timeout            |
| `backgroundTimeoutMs` | `3600000`               | Automatic-memory and dream timeout            |
| `shutdownTimeoutMs`   | `5000`                  | Best-effort Gateway shutdown drain budget     |

Configuration changes apply to subsequent runs. Failed automatic-memory batches are retained for retry, and the
Gateway shutdown hook attempts to flush pending work within `shutdownTimeoutMs`.

## Source and license

ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) and released under Apache-2.0.
