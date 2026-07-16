# ReMe memory provider for Hermes Agent

This plugin connects Hermes Agent to a running ReMe HTTP service. It recalls
relevant memory before each model call and records each completed turn through
ReMe's automatic memory job.

## Prerequisites

- Python 3.11 or newer
- A working Hermes Agent installation
- ReMe installed with its core dependencies
- One ReMe workspace and endpoint for each Hermes profile that should remain
  isolated

ReMe search currently covers one whole workspace. Pointing multiple Hermes
profiles at the same ReMe workspace therefore shares their recalled memory. Use
a separate ReMe workspace and endpoint when profiles must be isolated.

## Start ReMe

Start the HTTP service against a workspace dedicated to the active Hermes
profile:

```bash
reme start \
  workspace_dir="$HOME/.reme-hermes-default" \
  service.backend=http \
  service.host=127.0.0.1 \
  service.port=2333
```

ReMe needs a working LLM configuration for automatic memory extraction. Its
default search uses BM25, so embedding credentials are optional unless vector
retrieval is enabled. Keep the service running while Hermes is active.

## Install and configure

Hermes supports installing a plugin from a repository subdirectory:

```bash
hermes plugins install agentscope-ai/ReMe/plugins/hermes_agent
hermes memory setup
```

Select `reme`, accept `http://127.0.0.1:2333` or enter the endpoint used above.
Setup calls ReMe `health_check` and only replaces an existing provider config
after the endpoint reports healthy. Then start a new Hermes session.
Configuration is stored in
`$HERMES_HOME/reme.json`, so every Hermes profile can point to its own ReMe
workspace.

The file supports these optional settings:

```json
{
  "endpoint": "http://127.0.0.1:2333",
  "request_timeout": 600.0,
  "health_timeout": 2.0,
  "health_retry_seconds": 30.0,
  "recall_limit": 5
}
```

Run `hermes memory status` to check that the provider is installed and
configured. Starting a Hermes session performs a fresh endpoint health check.

## Lifecycle and failure behavior

- `prefetch` calls ReMe `search` and returns only its recalled text. Hermes wraps
  that text in its protected memory-context block.
- `sync_turn` sends the completed user/assistant turn to ReMe `auto_memory` with
  a filename-safe ID derived from the Hermes profile and conversation.
- Cron, flush, and subagent contexts do not write conversational memory.
- A failed health check disables recall and recording until the retry cooldown
  expires. Recall returns no context; a skipped or failed recording is logged as
  a warning instead of blocking the Hermes conversation loop.
- `shutdown` releases provider state; ReMe remains an independently managed
  service.
