---
title: Services and Deployment
description: Run ReMe through HTTP, SSE, MCP, the CLI, and ReMe Studio while respecting its local security boundary.
---

# Services and Deployment

ReMe can run as a local HTTP service, a standalone MCP server, or a one-shot CLI Job. By default HTTP binds to
`127.0.0.1:2333`; one process serves JSON, SSE, streamable HTTP MCP, and optional ReMe Studio.

## HTTP API

Start the service in one terminal and leave it running. Choose one of these startup commands:

```bash
reme start
reme start service.host=127.0.0.1 service.port=8181
```

From another terminal, call a Job through the CLI client:

```bash
reme search query="user preferences" limit=5
```

Regular Jobs also become `POST /<job-name>`; this HTTP example uses the default port `2333`:

```bash
curl -s http://127.0.0.1:2333/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"user preferences","limit":5}'
```

Job arguments live at the request body's top level. A regular response follows the `Response` schema:

```json
{"success": true, "answer": "...", "metadata": {}}
```

Unhandled Step failures become unsuccessful responses.

## Streaming and SSE

Jobs with `backend: stream` also use `POST /<job-name>`, returning `text/event-stream`. Error paths emit an error chunk and always terminate the stream. The default `chat` Job is streaming:

```bash
curl -N http://127.0.0.1:2333/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize my long-term preferences"}'
```

Studio shares one conversation UI between **Chat** and **Proactive Chat**. The default configuration exposes only
`chat`, with `query` and an optional `session_id` for independent conversations. Start with `config=proactive` to expose
the three `pa_*` Jobs instead of `chat`. Entries are enabled according to the service's exposed Jobs.

Proactive Chat uses the application-owned `pa_chat` stream with only `query`. `pa_session` takes no parameters and returns
`{"session_id": "...", "status": "idle"}` in `answer` (both values are null before a session exists). `pa_new_session` takes
no parameters, creates and selects an empty session, and returns its ID in `answer` without model generation.

Start the service with `reme start config=proactive` in one terminal, then use another terminal:

```bash
reme pa_session
reme pa_chat query="What should I follow up on?"
reme pa_new_session
```

The current session is shared within one Application and resets on restart; session files remain on disk. Chat and
session creation reject concurrent operations. Status queries do not wait for generation. PA Jobs currently require
`CodexAgentWrapper`; other wrappers raise `NotImplementedError`.
If the current native session is unavailable, `pa_session` returns `status: "error"`; create a new session to recover.
Once empty-session creation starts, selecting that session completes before cancellation is propagated.

Each new PA session loads workspace-relative `personal/REME_MEMORY.md` into its configured prompt. Missing or empty
files instruct the agent to naturally ask about preferences, current situation, interests, and projects in its first
reply, then create the file from confirmed information. The agent is instructed to read the latest file before updates
and keep it concise, complete, and current. PA persona, memory, onboarding, and forgetting prompts are loaded by
`PromptHandler` from `reme/steps/proactive/pa.yaml`, with English defaults and Chinese variants. The proactive config
defaults to `language=zh`; use `language=en` for English or a Step's `language` to override the application language.
Customize localized keys through the Step `prompt_dict` (for example, `system_prompt_zh`). Loaded memory text
is preserved. Explicit requests to forget information instruct the agent to
delete the matching content without copying it elsewhere or restoring it from old conversations or snapshots.
Maintenance also removes confirmed obsolete state, superseded preferences, abandoned interests, and finished project
details without lasting value, while retaining valid long-term facts. Creating a session itself does not generate a
reply or write memory.
The prompt snapshot is reused for that session; new sessions reload the file. Reads must be complete and at most
50 KiB; read errors or workspace-escaping paths fail session creation without replacing the current selection.

The `proactive` config also starts `pa_loop`: wait five minutes after startup or restart, discover interesting resources,
then decide whether the current PA conversation would benefit from a reminder. Both roles and their bilingual prompts
live in `reme/steps/proactive/`. Background rounds require a nonempty `personal/REME_MEMORY.md`; missing or unreadable
memory fails the round before agent calls or resource acknowledgements, then retries after five minutes.
The discovery agent calls `resource_changes` and `resource_processed` itself: acknowledge unrelated packages after assessment,
relevant packages after saving their findings, and leave failed packages pending. Notes live at
`{daily_dir}/YYYY-MM-DD/interest_HHMMSS.md`, with a numeric suffix for same-second collisions. Frontmatter `status` is
`unsent`, `ready`, `sent`, or `discarded`; code supplies IDs and timestamps and repairs notes with missing metadata.
Discovery may create only its requested output path and must preserve existing notes. The push role may edit only
`status` and `message` on existing candidates; invalid edits fail the round.
The push agent considers candidates across all dates, including rounds without new findings, with the current session
ID, native session file path, activity, and recent messages. At most one note may be `ready`, with reminder text in
frontmatter `message`. Code rechecks the target session and conversation turn before delivery, recovers interrupted
delivery with a stable message ID, rejects concurrent note changes, and records `target_session_id`, `sent_at`, and related
metadata. `sent` means appended
to the conversation, not read by the user.
The push agent may call private Job `set_next_wakeup(minutes=...)`, communicating through `ApplicationContext.metadata`.
The override applies once after the current round; omission, failure, and restart restore five minutes. `pa_proactive`
and `set_next_wakeup` are not exposed through the HTTP/MCP service; the latter is bound to the background agent's tools.
Disable background reminders with `jobs.pa_loop.enabled=false`.

MCP does not expose Stream Jobs.

## MCP

ReMe 0.4.2.2 uses FastMCP 4 and MCP Python SDK 2. Upgrade ReMe and host-side MCP bridges together;
standalone bridge environments require `fastmcp>=4.1.4,<5`. CLI tool catalogs use SDK 2 snake_case fields,
including `input_schema`, `output_schema`, and annotation hints. Update consumers that read the old camelCase fields.

The default HTTP service mounts streamable HTTP MCP at `/mcp`:

```yaml
service:
  backend: http
  mcp_enabled: true
  mcp_path: /mcp
  mcp_stateless_http: false
```

For a standalone MCP service:

```bash
reme start service.backend=mcp service.transport=stdio
reme start service.backend=mcp service.transport=sse service.port=2333
reme start service.backend=mcp service.transport=streamable-http service.port=2333
```

MCP tools come from non-stream Jobs with `enable_serve: true`. Use `service.jobs` as an allowlist. `injected_job_kwargs` adds server-managed values that callers cannot override.

```yaml
service:
  backend: http
  jobs: [search, read, traverse, auto_memory]
  injected_job_kwargs:
    tenant_id: local-user
  tool_error_on_failure: true
```

## ReMe Studio

After installing `reme-ai[web]` or `reme-ai[core]`, the default HTTP origin also serves Studio:

```text
http://127.0.0.1:2333/
```

Disable it with `service.web_enabled=false` or select a build with `service.web_static_dir`. Missing static assets do not prevent the Job API from starting.

## One-shot Jobs

```bash
reme start job=search query="user preferences" limit=5
```

This selects the one-shot CLI Service while retaining the normal Application, Component, and Job lifecycle.

## Service discovery

```bash
reme find_reme
```

Ordinary `reme <action>` commands prefer the running service's actual backend, host, port, and transport. They fall back to local configuration only when no service is discovered.

## Security boundary

ReMe is local-first:

- the default binds to `127.0.0.1`;
- HTTP CORS allows any origin;
- Jobs may write, move, or delete files;
- the service layer has no general-purpose user authentication.

Remote access must be enabled explicitly with `reme start service.host=0.0.0.0`. Do not expose the service directly to
the public internet. Place it on a controlled network or behind an authenticated TLS reverse proxy, apply access
controls and request-size limits, and expose only necessary Jobs.

## OpenAPI

FastAPI exposes the active endpoints through `/docs`, `/redoc`, and `/openapi.json`. The Studio SPA fallback preserves these reserved paths.
