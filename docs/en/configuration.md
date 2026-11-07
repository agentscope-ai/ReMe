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

Choose one startup command in one terminal and leave the service running:

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

From another terminal, call `reme <job-name>`, such as `reme help`; the CLI discovers the running service configuration.

## CLI values

Arguments use `key=value`; leading `-` or `--` is accepted:

```bash
reme start --service.port=8181 --service.web_enabled=false
```

Values support null, booleans, numbers, JSON arrays and objects, quoted JSON strings, and plain strings. Numeric-looking values with leading zeroes, such as `007`, remain strings, as do `nan`, `inf`, and `Infinity`. Quote values such as `"true"` in JSON when they must remain strings.

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
| `job_history_limit` | `100` | Completed executions retained per Job in application memory; running records are always retained |
| `components` | defaults | Components grouped by type and name |

`session_dir` must remain workspace-relative.

## Enabling jobs

`jobs.<name>.enabled` defaults to `true` for all Job types. Disabled jobs retain their configuration but do not
start, expose service interfaces, or accept calls through `Application.run_job()` / `run_stream_job()`.
For example, disable ReMe's Dream cron when a host plugin owns the schedule:

```bash
reme start jobs.dream_cron.enabled=false
```

Restart the service to apply the override. The separate `auto_dream` API remains available, and other jobs continue running.
`enable_serve` independently controls service exposure: `enabled=true, enable_serve=false` keeps a Job available for
local calls. Background and cron jobs are never service-exposed.

`jobs.<name>.track_execution` defaults to `true` for background/cron Jobs and `false` for base/stream Jobs.
Set it explicitly to override either default, for example `reme start jobs.chat.track_execution=true`.
Execution records are memory-only and disappear when the Application is recreated. `job_history_limit=0` retains only
running executions. See [Job execution status](./operations.md#job-execution-status) for queries and step paths.

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

## Web search

With `reme-ai[core]` installed, `dashscope_search` calls `qwen3.8-flash` through the DashScope SDK, independently of LLM settings.
Set the key and start the service in one terminal:

```bash
export DASHSCOPE_API_KEY=sk-xxx
reme start config=default
```

From another terminal, call `reme dashscope_search query="Latest Qwen models"`.
A missing key does not prevent service startup; calls to this Job return a failure.
The result's `search_info.search_results` preserves provider source fields; `content` is separate model-generated text.
Page bodies and snippets are not guaranteed. See `reme/config/default.yaml` for the model, search strategy, and timeout.

## Speech components

`asr` and `tts` are independent Components with abstract `BaseASR` and `BaseTTS` contracts under
`reme/components/`. The `dashscope` backends use the native asynchronous HTTP APIs and the same
`DASHSCOPE_API_KEY` as web search. Missing credentials fail a speech call without preventing startup.
Configure `base_url` with a native endpoint ending in `/api/v1`, rather than an OpenAI-compatible URL.

`reme/config/proactive.yaml` configures `qwen-audio-3.1-asr-flash` and `qwen-audio-3.1-tts-plus`
with voice `longanhuan_v3.1`. Start the service in one terminal:

```bash
export DASHSCOPE_API_KEY=sk-xxx
reme start config=proactive
```

From another terminal, test text → speech → text:

```bash
reme speech_demo text="How is the weather today?"
```

The Job runs `tts_step`, then `asr_step`, sharing audio bytes in the request context. Its answer is
the recognized text, with each provider's request ID and usage in response metadata. `tts_demo` runs
synthesis alone and returns Base64 audio, format, and sample rate; `asr_demo` accepts Base64 `audio`
with `audio_format` and optional `sample_rate`.

For direct component use, `transcribe(audio, audio_format=..., sample_rate=...)` returns an `ASRResult`.
`synthesize(text)` collects a `TTSResult`; `stream(text)` yields incremental `TTSResult` fragments.
Concatenate their `audio` bytes in order and require a final `is_last=True` result. Close an iterator
when stopping early. The result schemas live in `reme/schema/speech.py`. Microphone capture and
playback belong to the caller.

## Decision Model Configuration

`as_decision_model` is a provider-independent interface for probabilistic decisions. The current `agentscope` adapter
calls the TypeSafe API through AgentScope `2.0.11`. It evaluates named binary, choice, or score questions against shared text or a JSON object.

Install the optional dependencies:

```bash
pip install "reme-ai[core,decision]"
# For a source checkout:
pip install -e ".[core,decision]"
```

`core` excludes the decision model SDK; `full` includes this extra. Environment variables alone do not enable decision models, and
the default decision model configuration remains commented out. Startup does not import optional decision model modules or
create a client. The first decision request checks dependencies and credentials; shutdown waits for active requests before
closing the shared client.

| Environment variable | Default | Purpose |
|---|---|---|
| `DECISION_MODEL_BACKEND` | `agentscope` | Decision model backend |
| `DECISION_MODEL_NAME` | `null` (adapter default) | Model name or alias |
| `DECISION_MODEL_API_KEY` | Empty | TypeSafe API key, required for decision requests |
| `DECISION_MODEL_BASE_URL` | `null` | Optional API URL; empty values use the SDK default |

To enable a component:

```yaml
components:
  as_decision_model:
    default:
      backend: ${DECISION_MODEL_BACKEND:-agentscope}
      model: ${DECISION_MODEL_NAME:-null}
      credential:
        api_key: ${DECISION_MODEL_API_KEY:-}
        base_url: ${DECISION_MODEL_BASE_URL:-null}
      timeout: 30
      max_retries: 2
      retry_delay: 0.5
```

`timeout` is the per-request timeout in seconds; `max_retries` counts retries after the initial request; `retry_delay`
is the initial backoff in seconds. Configure these fields through YAML or CLI overrides. The SDK performs retries.

The `decision_demo` configuration uses `extends: default` to inherit the HTTP/MCP service, components, and Jobs,
then adds a decision model and a service-exposed `decide` Job. The default background tasks also start. First start the service
in one terminal:

```bash
export DECISION_MODEL_API_KEY=your-typesafe-api-key
reme start config=decision_demo \
  components.as_decision_model.default.timeout=15
```

Then call it from another terminal; the client discovers the running service configuration:

```bash
reme decide state="My account was charged twice. Please help urgently."
```

Calls send the state and questions to TypeSafe. Omitting `questions` evaluates urgency, customer-support routing, and
priority together. You can replace the demo questions completely and pass a JSON object as state:

```bash
reme decide \
  state='{"request":"My account was charged twice","channel":"email"}' \
  questions='{"urgent":{"type":"binary","instructions":"Does this need urgent attention?"}}'
```

Question types are `binary`, `choice`, and `score`. Binary questions accept optional `criteria.true` / `criteria.false`
descriptions. Choice questions require a non-empty `criteria` object mapping category names to descriptions or `null`.
Score questions require a non-empty ordered list of descriptions, starting at score zero. Each question can include
`instructions`. Explicit empty questions fail validation instead of selecting the demo defaults.

`reme decide` displays the standard Response JSON, with the decision result under `answer`, including `model`, `content`,
`usage`, response identity, and timing information. Answers preserve probabilities and
confidence and are keyed by question name; integer score-distribution keys become strings in JSON. The Step also writes
`context.data["decision_result"]` for subsequent Steps in that request, without automatically writing memory files.
SDK errors follow BaseJob's `success=false` contract.

Use `decision_demo_step` in your own Job and select a named component with the Step setting `as_decision_model: <name>`.
Implement `BaseAsDecisionModel` and register a backend to add another provider. Jobs select a component by name;
they do not select a provider directly. The current adapter uses AgentScope classifier question and answer schemas.
See `reme/components/as_decision_model/` for the interface. A null `model` uses the adapter default.

For tool-history compaction, start `reme start config=decision_compaction` and call `reme compact_context` from another
terminal with `messages` and any protection or budget options. It uses the same decision model interface and returns
a proposed snapshot; the caller owns archiving and applying it. Request token budgets are estimates with overhead
headroom, independent of the host model tokenizer.

See [Context compaction with decision models](./context_compaction.md) for the strategy, SVG walkthrough, and live integration cases.

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
