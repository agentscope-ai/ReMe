# ReMe for TypeScript agents

[中文说明](./README_ZH.md)

`@agentscope-ai/reme` connects DeepSeek Harness and OpenClaw to ReMe's local-first, file-native long-term memory. The package also exposes a host-independent ReMe HTTP client.

![ReMe Status in DeepSeek Harness](./figures/dsh/reme-status-overview.png)

## Capabilities

- Injects memory guidance and provides explicit `reme_search` lookup.
- Captures completed conversations through background `auto_memory` batches.
- Runs optional daily `auto_dream` consolidation in the workspace timezone.
- Keeps durable memory in user-owned `daily` and `digest` Markdown files.
- Uses each host's native lifecycle, tools, settings, and shutdown hooks.
- Excludes plugin context and tool results from automatic memory capture.

## Documentation

| Host             | English                     | 中文                                 |
| ---------------- | --------------------------- | ------------------------------------ |
| DeepSeek Harness | [Guide](./docs/dsh.md)      | [使用指南](./docs/dsh.zh-CN.md)      |
| OpenClaw         | [Guide](./docs/openclaw.md) | [使用指南](./docs/openclaw.zh-CN.md) |

## Quick start

Start ReMe:

```bash
pip install "reme-ai[core]"
reme start workspace_dir=/absolute/path/to/workspace
```

Install the adapter for your host:

```bash
# DeepSeek Harness
dsh plugin --profile web add @agentscope-ai/reme

# OpenClaw
openclaw plugins install clawhub:@agentscope-ai/reme
```

The default endpoint is `http://127.0.0.1:2333`. ReMe HTTP does not use API-key authentication, so keep it on loopback or another trusted network unless it is protected by a proxy.

## Client library

```ts
import { ReMeClient, formatReMeContext } from "@agentscope-ai/reme";
```

Host adapters are exported from `@agentscope-ai/reme/dsh` and `@agentscope-ai/reme/openclaw`. See the host guides for requirements, configuration, screenshots, troubleshooting, and release behavior.
