# ReMe for TypeScript agents

[中文说明](./README_ZH.md)

`@agentscope-ai/reme` provides one shared ReMe HTTP client and host adapters for DeepSeek Harness and OpenClaw. Each
adapter uses its host's native lifecycle and tool interfaces; importing the root package does not load either host.

The package expects a running ReMe HTTP service with the `search`, `auto_memory`, and `auto_dream` jobs required by the
selected adapter:

```bash
reme start workspace_dir=/absolute/path/to/workspace
```

The default endpoint is `http://127.0.0.1:2333`. All entries support `REME_URL`, or `REME_HOST` plus `REME_PORT`.
ReMe's HTTP service does not use API-key authentication.

## Requirements

- A running ReMe HTTP service with the jobs required by the selected adapter.
- Node.js `22.22.3+`, `24.15.0+`, or `25.9.0+` on the corresponding supported major-version line.
- DeepSeek Harness or OpenClaw only when using that host-specific entry; the root client has no host runtime dependency.

## DeepSeek Harness

Install the package as a DSH profile bundle:

```bash
dsh plugin --profile web add @agentscope-ai/reme
```

The bundle loads `@agentscope-ai/reme/dsh` in an isolated `remeMemory` realm. It injects durable memory guidance,
registers `reme_search`, submits completed main-agent turns to `auto_memory`, and runs the optional daily `auto_dream`
schedule. Recalled plugin context and tool results are excluded from automatic memory capture.

Configure the bundle by replacing its row in the profile's `cordis.patch.yml`:

```yaml
- id: reme-memory
  config:
    - id: reme-memory-runtime
      name: "@agentscope-ai/reme/dsh"
      config:
        endpoint: http://127.0.0.1:2333
        language: zh
        timezone: Asia/Shanghai
        autoMemoryInterval: 5
        autoDreamEnabled: true
        dreamCron: "0 23 * * *"
```

On the DSH Web profile, the same fields are available under **Settings → Plugins → Plugin configuration → ReMe
Memory**. Changes are stored in DSH's user settings document and apply to subsequent requests and captures. Changing
the daily dream controls reschedules the next run; changing the guidance language affects newly started sessions.
The test-only `dreamIntervalMs` value remains outside the user-settings section.

| Option                | Default                 | Meaning                                       |
| --------------------- | ----------------------- | --------------------------------------------- |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP service URL                         |
| `language`            | `en`                    | Memory guidance language: `en` or `zh`        |
| `autoMemoryEnabled`   | `true`                  | Capture completed main-agent turns            |
| `autoMemoryInterval`  | `5`                     | Submit after this many completed turns        |
| `autoDreamEnabled`    | `true`                  | Enable daily dream maintenance                |
| `dreamCron`           | `0 23 * * *`            | Daily schedule in the workspace timezone      |
| `dreamHint`           | empty                   | Optional guidance sent to `auto_dream`        |
| `rootAgentsOnly`      | `true`                  | Exclude subagents from guidance and capture   |
| `searchLimit`         | `5`                     | Maximum results returned by `reme_search`     |
| `requestTimeoutMs`    | `10000`                 | Search request timeout                        |
| `backgroundTimeoutMs` | `3600000`               | Automatic-memory and dream timeout            |
| `shutdownTimeoutMs`   | `5000`                  | Best-effort shutdown drain budget             |
| `timezone`            | `Asia/Shanghai`         | IANA timezone used for batches and scheduling |

The ReMe card reads the service's `health_check` and `status` jobs on demand. It shows the ReMe version, component
health, chunk/index counts, process RSS, and estimated component memory; it can also display the redacted `app_config`
response and trigger one `auto_dream` run. Diagnostics are refreshed when the card first opens or when the user asks,
not polled continuously. The page calls the configured ReMe HTTP endpoint from the local browser, so that service must
remain browser-reachable and allow the DSH origin.

## OpenClaw

OpenClaw `2026.7.1` or later can install the same package. The current SDK and
OpenClaw Gateway require Node.js `22.22.3+`, `24.15.0+`, or `25.9.0+` on their
respective major-version lines:

```bash
openclaw plugins install @agentscope-ai/reme
```

Select `reme` for `plugins.slots.memory` when another memory plugin is active. The adapter uses OpenClaw's current
`before_prompt_build` hook, registers the `reme_search` action, injects durable memory guidance, and recalls relevant
memory before conversational root-agent runs. Completed user/assistant pairs are grouped into per-session,
date-consistent batches for `auto_memory`; failed batches are retained for retry and pending work is flushed within a
bounded Gateway shutdown budget. One plugin-owned daily schedule runs `auto_dream`. Recall is wrapped in
`<reme-context>` and marked as untrusted historical data. Cron, heartbeat, memory, overflow, and subagent runs do not
recall or capture conversational memory by default.

OpenClaw plugin configuration accepts:

| Option                | Default                 | Meaning                                       |
| --------------------- | ----------------------- | --------------------------------------------- |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP service URL                         |
| `language`            | `en`                    | Memory guidance language: `en` or `zh`        |
| `autoRecall`          | `true`                  | Recall before conversational root-agent runs  |
| `searchLimit`         | `5`                     | Maximum search results                        |
| `recallMinScore`      | `0`                     | Minimum search score                          |
| `autoMemoryEnabled`   | `true`                  | Capture completed conversational turns        |
| `autoMemoryInterval`  | `5`                     | Submit after this many completed turns        |
| `autoDreamEnabled`    | `true`                  | Enable daily memory consolidation             |
| `dreamCron`           | `0 23 * * *`            | Daily schedule in the workspace timezone      |
| `dreamHint`           | empty                   | Optional guidance sent to `auto_dream`        |
| `rootAgentsOnly`      | `true`                  | Exclude subagents from guidance and capture   |
| `timezone`            | `Asia/Shanghai`         | IANA timezone used for batches and scheduling |
| `requestTimeoutMs`    | `10000`                 | Recall and explicit search timeout            |
| `backgroundTimeoutMs` | `3600000`               | Automatic-memory and dream timeout            |
| `shutdownTimeoutMs`   | `5000`                  | Best-effort shutdown drain budget             |

OpenClaw's conversation-access and prompt-injection permissions remain host settings; enable them for ReMe when your
OpenClaw policy requires explicit grants. The adapter does not modify Gateway configuration.

## Library entry

Consumers that only need the transport can import the root package:

```ts
import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";
```

Host code is available only through `@agentscope-ai/reme/dsh` and `@agentscope-ai/reme/openclaw`.

## Development

```bash
cd typescript
npm ci
npm run format:check
npm run lint
npm run typecheck
npm test
npm run test:package
```

`npm pack` runs the TypeScript build and includes only `dist`, the DSH patch, the OpenClaw manifest, and the English and
Chinese READMEs.

Stable releases use npm's `latest` distribution tag; prereleases use `next`.
