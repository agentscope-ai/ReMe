---
name: reme_memory
description: Set up and use ReMe as a file-native long-term memory system through the reme CLI. Use when an Agent needs to detect whether ReMe is installed or running, install and configure ReMe, start or verify its local service, retrieve prior context, or write and consolidate durable memory.
---

# ReMe Memory

Use ReMe as the persistent memory layer for this Agent. ReMe stores raw sessions, daily notes, resources, and long-term
digest memories in a user-owned local workspace.

## Bootstrap ReMe

Run this workflow before first use and whenever a ReMe command cannot reach the service. Distinguish a missing CLI from
an installed but stopped service.

### 1. Check whether ReMe is installed

Run:

```bash
command -v reme
```

If this prints an executable path, treat ReMe as installed and continue to service discovery. Do not reinstall or upgrade
an existing installation unless the user requests it.

If the command is missing, check Python before installing:

```bash
python3 -c 'import sys; print(sys.version); raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
```

ReMe requires Python 3.11 or newer. If the user has requested setup or installation, install the recommended package in
the active Python environment:

```bash
python3 -m pip install "reme-ai[core]"
```

When working from a ReMe source checkout and the user explicitly wants an editable source installation, run this from
the repository root instead:

```bash
python3 -m pip install -e ".[core]"
```

Do not silently install into or modify a Python environment when the user only asked to use memory. Explain that ReMe is
missing and ask before installing. After installation, run `command -v reme` again. If it is still missing, check that
the active environment's executable directory is on `PATH`; do not repeatedly reinstall.

### 2. Configure optional model credentials

Basic file operations, BM25 search, wikilink traversal, and reading existing proactive topics work without model
credentials. `auto_memory`, `auto_resource`, and `auto_dream` require an LLM configuration.

When those model-powered jobs are needed, have the user provide valid values through the environment or a `.env` file:

```dotenv
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

The default LLM backend is OpenAI-compatible and the default model is `qwen3.7-plus`. Override them when the endpoint
requires different values:

```dotenv
LLM_BACKEND=openai
LLM_MODEL_NAME=qwen3.7-plus
```

ReMe searches for `.env` in the directory where its command starts and up to five parent directories. Start the service
from a stable directory where the intended `.env` is discoverable. Never expose, log, or commit credentials.

Embedding retrieval is disabled by default. Do not request `EMBEDDING_API_KEY` merely to use the default BM25 and
wikilink search. Enabling vector retrieval also requires changing the embedding components in ReMe's configuration; do
not claim that setting an embedding key alone enables it.

### 3. Discover or start the service

Check for an existing ReMe service before starting another one:

```bash
reme find_reme
```

If it prints `HOST=... PORT=... PID=...`, reuse that service and its workspace. Do not start a duplicate or change its
workspace configuration.

If it reports `reme not started`, start ReMe in a persistent terminal or managed process and leave it running:

```bash
reme start
```

The default HTTP address is `127.0.0.1:2333`, and the default workspace is `.reme/` under the startup directory. For
durable Agent memory, prefer a stable, user-selected workspace path so memory does not depend on the caller's current
directory:

```bash
reme start workspace_dir="/absolute/path/to/reme-workspace"
```

If port `2333` is occupied, do not stop or replace the unknown listener. Start ReMe on another port:

```bash
reme start workspace_dir="/absolute/path/to/reme-workspace" service.port=8181
```

Keep the startup command and workspace choice consistent across restarts. ReMe CLI commands discover a locally running
ReMe process, including one started with a custom port.

### 4. Verify readiness

After the service starts, run these commands from another terminal or tool session:

```bash
reme find_reme
reme version
reme health_check
```

Proceed only when `version` responds and `health_check` reports a healthy service. Use `reme list` to inspect the jobs
exposed by the running configuration. If verification fails, report the exact error and keep installation failure,
service discovery failure, port conflict, and missing model credentials as separate diagnoses.

## Retrieve Memory

Before answering questions about previous conversations, user preferences, project history, decisions, resources, or
long-term context, search ReMe first:

```bash
reme search query="<question or keywords>" limit=5
```

Read the relevant result rather than relying only on the search snippet:

```bash
reme read path="<workspace-relative-path>"
reme read path="<workspace-relative-path>" start_line=1 end_line=80
```

Use `traverse` when wikilink neighbors may matter:

```bash
reme traverse path="<workspace-relative-path>" depth=1 direction=both
```

Cite the workspace-relative paths used. If retrieval returns nothing useful, say so plainly instead of inventing prior
context.

## Write Memory

Record durable facts, user preferences, important decisions, project context, and lessons learned. Avoid secrets or
sensitive personal data unless the user explicitly asks to store them.

For an ordinary conversation, call `auto_memory` with the current messages and a stable session ID:

```bash
reme auto_memory \
  session_id="<session-id>" \
  messages='[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]' \
  memory_hint="<why this should be remembered>"
```

This job requires the LLM configuration described above. A missing LLM credential is not evidence that basic ReMe file
operations or BM25 retrieval are unavailable.

For explicit file operations, read before editing and preserve existing content unless replacement is intended:

```bash
reme write path="daily/<YYYY-MM-DD>/<name>.md" name="<name>" description="<description>" content="<markdown>"
reme edit path="<workspace-relative-path>" old="<old text>" new="<new text>"
```

Use ReMe commands instead of editing memory files directly unless the user explicitly asks for direct file maintenance.

## Ingest Resources

Place external documents under `resource/YYYY-MM-DD/` in the selected ReMe workspace. The default background watcher
processes supported new or changed files while `reme start` is running.

To request processing explicitly:

```bash
reme auto_resource changes='[{"path":"resource/<YYYY-MM-DD>/<file>","change":"added"}]'
```

`auto_resource` requires LLM credentials.

## Consolidate and Use Proactive Topics

The default service runs background and cron jobs while it remains active. `auto_dream` consolidates daily notes and
resource interpretations into long-term digest memory and generates interest topics. Run it manually when the host owns
the schedule or the user requests consolidation:

```bash
reme auto_dream date="<YYYY-MM-DD>"
```

Read generated topics with:

```bash
reme proactive date="<YYYY-MM-DD>"
```

`auto_dream` requires LLM credentials. `proactive` reads existing structured topics and works without an LLM call. Pass
`include_content=false` when raw YAML content is unnecessary. The host Agent decides whether and how to mention topics;
ReMe does not independently notify the user or take external action.

## Integration Rules

- Reuse a healthy running service; never start one ReMe process per command or conversation.
- Keep one stable workspace for contexts that should share memory. Use separate workspaces when profiles must be isolated.
- Call `auto_memory` after useful conversation turns only when the host owns lifecycle integration.
- Use ReMe's in-process `ReMe` Python API instead of the CLI when embedding it into a Python host application.
- Prefer the dedicated integrations under `plugins/claude_code/reme` and `plugins/hermes_agent` for those hosts.
- Treat user-owned memory files as source data. Do not delete, rewrite, or migrate a workspace merely to repair an index;
  use rebuildable index operations such as `reme reindex` when appropriate.
