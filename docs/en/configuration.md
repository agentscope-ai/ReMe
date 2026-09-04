---
title: Configuration
description: ReMe configuration files, environment expansion, command-line overrides, and core components.
---

# Configuration

ReMe uses YAML or JSON to describe its Service, Jobs, and Components. The built-in default is `reme/config/default.yaml`. Select another configuration at startup and apply command-line overrides when needed.

## Precedence

Configuration is merged in this order, with later values winning:

1. The selected file; `default` is used when none is specified.
2. `application_defaults` from enabled plugins.
3. CLI dot-notation overrides.

```bash
reme start
reme start config=demo
reme start config=/absolute/path/to/app.yaml
reme start service.port=8181 workspace_dir=/data/reme
```

`config` accepts a built-in name or a `.yaml`, `.yml`, or `.json` file. Overrides are deep-merged, so changing `service.port` preserves sibling service settings.

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

File operations, BM25 search, and wikilink traversal do not require an LLM. Evolution workflows such as `auto_memory`, `auto_resource`, and `auto_dream` do.

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

## Inspect the effective configuration

```bash
reme app_config
```

The result is the merged, validated configuration with secrets redacted. Use it when diagnosing plugin or override precedence. The authoritative contracts remain `reme/schema/application_config.py` and `reme/config/default.yaml`.
