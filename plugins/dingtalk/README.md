# DingTalk Plugin

[中文](README_ZH.md)

DingTalk provides outbound Markdown notifications and a long-running Agent chat bridge for ReMe. This directory is an
independent Python distribution. Its `reme.plugins` entry point exposes a backend-only `plugin.yaml` that registers
`dingtalk_markdown_send_step` and `dingtalk_wait_step`. Applications decide whether and how to use those backends;
enabling the plugin alone neither exposes a sending endpoint nor starts a long-lived connection.

## Quick start

Install ReMe and the plugin:

```bash
python -m pip install "reme-ai[core]>=0.4.1.12"
reme plugins install plugins/dingtalk
reme plugins validate dingtalk
```

Set the credentials for an internal DingTalk app robot and the target group conversation IDs:

```dotenv
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

The repository's `cookbook` config composes DingTalk with the independently installed Auto Fin and Daily Paper plugins:

```bash
reme plugins install plugins/auto-fin
reme plugins install plugins/daily_paper
reme start config=cookbook
```

`cookbook.yaml` owns the cross-plugin routing: it adds notification Steps after automatic tagging, shares each pipeline
with its cron Job, and explicitly starts the DingTalk Agent bridge as a background Job. The business plugins remain
usable without DingTalk.

## Markdown sending Job example

Add a private one-shot Job to an application config when direct sending is useful:

```yaml
extends: default
plugins: [dingtalk]

jobs:
  dingtalk_send:
    backend: base
    enable_serve: false
    parameters:
      type: object
      properties:
        markdown_path:
          type: string
      required: [markdown_path]
    steps:
      - backend: dingtalk_markdown_send_step
        app_key: ${DINGTALK_APP_KEY}
        app_secret: ${DINGTALK_APP_SECRET}
        robot_code: ${DINGTALK_ROBOT_CODE}
        conversation_ids: ${DINGTALK_CONVERSATION_IDS:-}
        timeout: 15
```

Run it without exposing the Job over HTTP or MCP:

```bash
reme start config=/path/to/dingtalk.yaml \
  job=dingtalk_send \
  markdown_path=daily/2026-09-15/report.md
```

`markdown_path` must resolve to an existing `.md` file inside the configured workspace. Frontmatter is omitted from
the message; the Markdown body is sent serially to each non-empty, comma-separated conversation ID. If no conversation
IDs are configured, delivery succeeds as a no-op. Once recipients are configured, missing credentials, invalid paths,
empty documents, token failures, and failed recipients make the Job fail explicitly. Delivery metadata reports the
configured and successfully sent counts without logging credentials or conversation IDs.

## Agent bridge

The bridge maps each sender and conversation to an independent Agent session, supports ReMe's session commands, and
sends the final Agent response as Markdown. Add this background Job to an application config that extends the built-in
default configuration:

```yaml
extends: default
plugins: [dingtalk]

jobs:
  dingtalk_agent:
    backend: background
    supervisor: true
    close_timeout: 10
    steps:
      - backend: dingtalk_wait_step
        agent_wrapper: default
        app_key: ${DINGTALK_APP_KEY}
        app_secret: ${DINGTALK_APP_SECRET}
        robot_code: ${DINGTALK_ROBOT_CODE}
        worker_count: 4
        builtin_tools: false
        job_tools: [search, read]
```

The same background Job is enabled by `reme/config/cookbook.yaml`. When used independently, start it with
`reme start config=/path/to/dingtalk.yaml`. A background Job is never exposed through HTTP or MCP.
Messages from different session keys may run concurrently, while messages sharing a key are serialized. Session IDs
live only in `ApplicationContext.metadata`, so restarting ReMe starts new DingTalk-to-Agent session mappings. The
configured `agent_wrapper` controls model access and permissions; grant only the built-in and Job tools the bot needs.

## Development

From the repository root:

```bash
reme plugins install ./plugins/dingtalk --editable
reme plugins validate dingtalk
python -m pytest plugins/dingtalk -v
```

Tests mock DingTalk HTTP, WebSocket, and Agent boundaries and do not contact external services.
