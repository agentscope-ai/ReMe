# ReMe memory for OpenClaw

[中文说明](./README_ZH.md)

ReMe gives OpenClaw file-native long-term memory while keeping durable memory in a workspace you own. The integration
uses OpenClaw's public plugin SDK, memory slot, authenticated Control UI, service lifecycle, conversation hooks, and
tool protocol.

![ReMe status in the OpenClaw Control UI](./figures/status-overview.png)

## Capabilities

- `reme_search` performs an explicit, source-backed memory search.
- `before_prompt_build` recalls relevant memory before a conversational root-agent turn.
- `agent_end` captures the original user/assistant pair in serialized background batches.
- `session_end` flushes a session boundary; service shutdown drains all work within a bounded budget.
- Auto Dream runs once per configured day and consolidates daily notes into durable digest knowledge.
- The **ReMe Memory** sidebar tab shows health, capture settings, Auto Dream state, and component diagnostics.

Recalled text is wrapped in `<reme-context>`, marked as untrusted historical data, and escaped so memory content cannot
close the wrapper. Subagent, cron, heartbeat, memory, and overflow runs are excluded by default.

## Compatibility and prerequisites

- OpenClaw `2026.9.3` or newer.
- Node.js `24.16.0+` on Node 24, or `26.1.0+` on Node 26.
- Python 3.11+ and a ReMe HTTP service exposing `search`, `auto_memory`, `auto_dream`, `health_check`, and `status`.
- A model configured for ReMe's automatic-memory and Auto Dream jobs.

The plugin never receives an LLM API key. Model credentials belong to ReMe/OpenClaw configuration and must not be put
in this package, a screenshot, or a committed config file.

## 1. Start ReMe

Install ReMe, choose a user-owned workspace, and bind the service to loopback. Port `3458` is useful when another ReMe
instance already uses the default port.

```bash
pip install "reme-ai[core]"
reme start \
  workspace_dir=/absolute/path/to/reme-workspace \
  service.host=127.0.0.1 \
  service.port=3458
```

Verify the service without exposing configuration or credentials:

```bash
curl -fsS -X POST http://127.0.0.1:3458/health_check \
  -H 'Content-Type: application/json' -d '{}'
```

ReMe HTTP does not add API-key authentication. Keep it on loopback or place it behind a trusted authenticated proxy.

## 2. Install the plugin

From a released package:

```bash
openclaw plugins install clawhub:@agentscope-ai/reme-openclaw-plugin
```

From this repository, build and install the exact archive that was tested:

```bash
cd /path/to/ReMe/integrations/openclaw
npm ci
npm run build
npm pack
openclaw plugins install --force ./agentscope-ai-reme-openclaw-plugin-0.1.0.tgz
```

Restart the Gateway after installation. In **Settings → Plugins**, search for `ReMe`; it should be enabled, categorized
as Memory, and expose `reme_search`.

![Installed ReMe plugin](./figures/plugin-installed.png)

The plugin details also show the required conversation grant. No secret fields are part of the plugin schema.

![ReMe plugin grants and tool contract](./figures/plugin-configuration.png)

## 3. Configure OpenClaw

Add the following to the effective OpenClaw config, then restart the Gateway:

```json
{
  "plugins": {
    "slots": { "memory": "reme" },
    "entries": {
      "reme": {
        "enabled": true,
        "hooks": { "allowConversationAccess": true },
        "config": {
          "endpoint": "http://127.0.0.1:3458",
          "language": "en",
          "autoRecall": true,
          "autoMemoryEnabled": true,
          "autoMemoryInterval": 5,
          "autoDreamEnabled": true,
          "dreamCron": "0 23 * * *",
          "timezone": "Asia/Shanghai"
        }
      }
    }
  }
}
```

Both settings outside `config` are required: `plugins.slots.memory` makes ReMe the active memory provider, while
`allowConversationAccess` permits the non-bundled plugin to inspect completed conversation turns. Without the latter,
explicit search may work while automatic capture does not.

Validate and inspect the effective runtime:

```bash
openclaw config validate --json
openclaw plugins inspect reme --runtime --json
openclaw gateway status
```

Runtime inspection should report `status: loaded`, `memorySlotSelected: true`, `reme_search`, three typed hooks, one
service, and one authenticated HTTP route. Current `plugins validate` targets authoring-metadata-only tool/feature
plugins; use runtime inspection for this mixed lifecycle plugin.

## Configuration reference

| Option                | Default                 | Meaning                                           |
| --------------------- | ----------------------- | ------------------------------------------------- |
| `endpoint`            | `http://127.0.0.1:2333` | ReMe HTTP service URL                             |
| `language`            | `en`                    | Memory guidance language: `en` or `zh`            |
| `autoRecall`          | `true`                  | Recall before conversational root-agent turns     |
| `searchLimit`         | `5`                     | Maximum returned search results                   |
| `recallMinScore`      | `0`                     | Minimum score for automatic recall                |
| `autoMemoryEnabled`   | `true`                  | Capture completed user/assistant turns            |
| `autoMemoryInterval`  | `5`                     | Completed turns per capture batch                 |
| `autoDreamEnabled`    | `true`                  | Enable scheduled consolidation                    |
| `dreamCron`           | `0 23 * * *`            | Daily schedule (`minute hour * * *`)              |
| `dreamHint`           | empty                   | Optional guidance passed to `auto_dream`          |
| `rootAgentsOnly`      | `true`                  | Exclude subagents and non-conversational triggers |
| `timezone`            | `Asia/Shanghai`         | IANA timezone used for batching and scheduling    |
| `requestTimeoutMs`    | `10000`                 | Recall, search, and status timeout                |
| `backgroundTimeoutMs` | `3600000`               | Automatic-memory and Auto Dream timeout           |
| `shutdownTimeoutMs`   | `5000`                  | Best-effort Gateway shutdown drain budget         |

Failed capture batches are retained in memory for retry. Durable state is written only by ReMe into its configured
workspace; plugin queues and diagnostics are process-local and rebuildable.

## 4. Use and verify

### Status frontend

Open **ReMe Memory** in the sidebar. `/plugins/reme/status/` is protected by Gateway auth. The page is server-rendered
because OpenClaw embeds external plugin tabs in a script-free sandbox. It shows the endpoint and component metrics, but
not model credentials or conversation content.

### Explicit search

Ask: `Use reme_search to look up Project Lighthouse.` The transcript should show a **ReMe Search** tool invocation and
the answer should cite workspace-relative memory paths.

![Explicit reme_search result](./figures/memory-search.png)

### Automatic recall

Start a new session and ask a question existing memory can answer, adding `Without calling tools`. A correct answer
with no tool card proves that `before_prompt_build` injected recalled context.

![Automatic recall without a tool call](./figures/automatic-recall.png)

### Automatic memory across sessions

For a quick test, temporarily set `autoMemoryInterval` to `1`. Tell OpenClaw a unique, synthetic fact, wait for the turn
and ReMe background job to finish, then ask for it in a different session without calling tools. Also confirm that a
new Markdown note exists beneath the configured ReMe workspace.

![A newly captured fact recalled in another session](./figures/conversation-memory.png)

Restore a larger interval after testing to reduce model calls.

### Auto Dream

Wait for `dreamCron`, then reload **ReMe Memory → Memory Consolidation**, verify `Last result: completed`, and inspect
the ReMe digest update. The plugin status route is read-only; memory consolidation runs only through the configured
scheduler.

![Completed Auto Dream run](./figures/auto-dream.png)

## Troubleshooting

- **Plugin is loaded but automatic recall/capture is absent:** check the memory slot and
  `hooks.allowConversationAccess`; restart after changing either.
- **ReMe Memory says Offline:** call ReMe `health_check`, verify the port, and ensure both processes can reach the same
  loopback/network namespace.
- **Explicit search works but capture does not:** verify `autoMemoryEnabled`, use interval `1` for diagnosis, and check
  ReMe logs for `/auto_memory` requests.
- **Capture waits after a short conversation:** the default is five completed turns. Session end and Gateway shutdown
  also attempt a bounded flush.
- **Auto Dream never runs:** the accepted cron form is daily `minute hour * * *`; confirm the IANA timezone and next run
  shown in the status tab.
- **Gateway rejects the plugin:** use a supported Node/OpenClaw version, rebuild, then inspect runtime diagnostics.
- **Status tab is blank after an upgrade:** confirm runtime inspection reports one authenticated HTTP route and restart
  the Gateway. Do not relax iframe sandboxing; this integration is designed for it.

## Development checks

```bash
cd integrations/openclaw
npm ci
npm run format:check
npm run lint
npm run typecheck
npm test
npm run test:package
```

Tests mock network boundaries. Real E2E work should use a temporary ReMe workspace and synthetic data; never commit
`.env`, runtime sessions, indexes, logs, caches, packed archives, or generated test memory.

## Source and license

ReMe is developed at [agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe) and released under Apache-2.0.
