---
title: Configuration
description: ReMe configuration files, environment expansion, command-line overrides, and core components.
---

# Configuration

ReMe uses YAML or JSON to describe its Service, Jobs, and Components. The built-in default is `reme/config/default.yaml`. Select another configuration at startup and apply command-line overrides when needed.

## Precedence

Configuration is merged in this order, with later values winning:

1. `application_defaults` from enabled plugins.
2. The selected file; `default` is used when none is specified.
3. CLI dot-notation overrides.

```bash
reme start
reme start config=demo
reme start config=cookbook
reme start config=/absolute/path/to/app.yaml
reme start service.port=8181 workspace_dir=/data/reme
```

`config` accepts a built-in name or a `.yaml`, `.yml`, or `.json` file. Overrides are deep-merged, so changing `service.port` preserves sibling service settings.
The optional `cookbook` variant extends `default` and composes the separately installed Auto Fin, Daily Paper, and
DingTalk plugins. It requires the three DingTalk application credential environment variables before configuration
loading. It also enables `text-embedding-v4` vector retrieval, uses AgentScope with
`${LLM_MODEL_NAME:-qwen3.8-max}` by default, and runs the DingTalk bridge through Claude Code with the same
`LLM_MODEL_NAME` and `LLM_API_KEY`.

## CLI values

Arguments use `key=value`; leading `-` or `--` is accepted:

```bash
reme start --service.port=8181 --service.web_enabled=false
```

Values support null, booleans, numbers, JSON arrays and objects, quoted JSON strings, and plain strings. Numeric-looking values with leading zeroes, such as `007`, remain strings. Quote values such as `"true"` in JSON when they must remain strings.

## Environment variables

Configuration recursively expands:

```yaml
api_key: ${LLM_API_KEY}
base_url: ${LLM_BASE_URL:-https://example.com/v1}
```

`${VAR}` fails when undefined; `${VAR:-default}` uses its fallback. ReMe also searches for `.env` from the command's working directory through at most five parents.

Keep secrets in `.env` or the process environment, never in committed configuration.

## Application fields

| Field | Default | Purpose |
|---|---|---|
| `app_name` | `ReMe` | Display name |
| `workspace_dir` | `.reme` | User-owned workspace root, normalized to an absolute path |
| `metadata_dir` | `metadata` | Rebuildable indexes, graphs, and catalogs |
| `session_dir` | `session` | Agent sessions; standard transcripts use `session/dialog` |
| `mem_session_dir` | `mem_session` | Agent-wrapper sessions and configuration |
| `resource_dir` | `resource` | External resources |
| `daily_dir` | `daily` | Daily memory |
| `digest_dir` | `digest` | Consolidated long-term memory |
| `timezone` | `Asia/Shanghai` | IANA timezone used for dates and cron jobs |
| `language` | empty | Default language for LLM interactions |
| `plugins` | `[]` | Installed plugins enabled for this Application |
| `service` | HTTP | Service configuration |
| `jobs` | default Jobs | Job configurations by name |
| `components` | defaults | Components grouped by type and name |

`session_dir` must remain workspace-relative.

## LLM

The default LLM uses an OpenAI-compatible interface:

```yaml
components:
  as_llm:
    default:
      backend: openai
      model: qwen3.7-plus
      context_size: 200000
      credential:
        api_key: ${LLM_API_KEY:-}
        base_url: ${LLM_BASE_URL:-}
```

Built-in registrations include `openai`, `anthropic`, `dashscope`, `deepseek`, `gemini`, `moonshot`, `ollama`, and `xai`. Their detailed model fields follow the corresponding AgentScope wrappers.

File operations, BM25 search, wikilink traversal, and `proactive_read` do not require an LLM. Evolution workflows such
as `auto_memory`, `auto_resource`, `auto_dream`, and proactive refresh do.

## Embeddings

Vector retrieval is disabled by default. Credentials alone do not enable it: configure `as_embedding`, `embedding_store`, and connect the store to `file_store`.

```yaml
components:
  as_embedding:
    default:
      backend: openai
      model: text-embedding-v4
      dimensions: 1024
      credential:
        api_key: ${EMBEDDING_API_KEY}
        base_url: ${EMBEDDING_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
  embedding_store:
    default:
      backend: local
      as_embedding: default
  file_store:
    default:
      backend: local
      embedding_store: default
      keyword_index: default
      file_graph: default
```

Rebuild the embedding index after changing the model or dimensions.

## Service and Jobs

Minimal HTTP configuration:

```yaml
service:
  backend: http
  host: 127.0.0.1
  port: 2333
  web_enabled: true
  mcp_enabled: true
  mcp_path: /mcp
```

A Job declares a backend, parameter schema, and ordered Steps:

```yaml
jobs:
  example:
    backend: base
    description: Example job
    parameters:
      type: object
      properties:
        text: { type: string }
      required: [text]
    steps:
      - backend: example_step
```

Set `enable_serve: false` to keep a Job internal. Background and cron Jobs are never service-exposed.

### Disable a Job

Every Job backend supports `enabled`, which defaults to `true`. To disable a default Job, use a CLI override:

```bash
reme start jobs.dream_cron.enabled=false
```

Or set `enabled: false` in that Job's YAML definition. Disabling a Job skips its construction, Step/backend resolution,
startup, and service registration. Its configuration remains visible and schema-validated. Local calls, including
calls from other Steps and agent tool selection, fail with a named disabled-Job error. One-shot CLI execution exits
with status 1 before application startup. HTTP/MCP do not advertise the Job and reject calls through their normal
unavailable-route/tool behavior. An explicit `service.jobs` allowlist containing a disabled Job is a configuration error.

`enable_serve: false` only hides an enabled Job from HTTP/MCP; it does not disable automatic or local execution.
Activation is fixed when the Application is constructed: create a new Application after changing this setting.
An empty `jobs: {}` override does not remove default Jobs because configuration is deep-merged. Disabling a Job
does not disable its shared Components, unload its plugin, or disable other Jobs that use the same Steps.

Review the effects before disabling the built-in background and cron Jobs:

| Job | Work and state | Effect of disabling |
|---|---|---|
| `index_update_loop` | Initial scan and ongoing updates of local file, keyword, vector, graph, and tag indexes; in-memory state with derived checkpoints | Local search can be missing or stale, even if another process updates its own index |
| `resource_watch_loop` | Resource catalog updates and automatic resource interpretation into daily notes | New/changed resources are not processed automatically |
| `digest_watch_loop` | Daily/digest catalog refresh and change logging | That catalog and its change log stop refreshing; this loop does not itself run dream consolidation |
| `dream_cron` | Scheduled extraction/integration into digest files, dream catalog checkpoints, and tagging | Scheduled dream work stops; separately enabled `auto_dream` remains callable |
| `optimize_index_cron` | Backend-specific maintenance of the local derived index | Scheduled index optimization stops |
| `proactive_refresh_cron` | Generates proactive interests/agenda files and updates the proactive catalog | Scheduled refresh stops; separately enabled `proactive_refresh` remains callable |

Files may be on a shared filesystem, but local indexes and catalogs retain process-local state. Keep each process's
required refresh Jobs enabled. Selecting one process to run a writer is an operator convention, not leader election,
exactly-once execution, or evidence of replica consistency. This switch does not provide cluster deployment support.

## Inspect the effective configuration

```bash
reme app_config
```

The result is the merged, validated configuration with secrets redacted. Use it when diagnosing plugin or override precedence. The authoritative contracts remain `reme/schema/application_config.py` and `reme/config/default.yaml`.
