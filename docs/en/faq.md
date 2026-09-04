---
title: Frequently Asked Questions
description: Quick answers for ReMe installation, services, models, retrieval, files, and plugins.
---

# Frequently Asked Questions

## Do basic file operations require a model API key?

No. `write`, `read`, `list`, `stat`, BM25 search, and wikilink traversal work without model credentials. `auto_memory`, `auto_resource`, and `auto_dream` require an LLM.

## Why is search still BM25-only after setting an embedding key?

Embeddings are disabled by default. Configure `as_embedding` and `embedding_store`, then connect `file_store.default.embedding_store` to that component. See [Configuration](./configuration.md#embeddings).

## Why did `reme reindex` not discover a new file?

`reindex` rebuilds indexes from current `file_chunks`; it does not scan the workspace. Check `index_update_loop`, the watched directory and extension, and `health_check`.

## How do I use another workspace?

```bash
reme start workspace_dir=/absolute/path/to/memory
```

Ordinary CLI calls discover the running service, so they do not need the workspace argument again.

## What if port 2333 is occupied?

Do not stop an unknown listener. Select another port:

```bash
reme start service.port=8181
```

Then confirm it with `reme find_reme`.

## Why is an installed plugin missing its Jobs?

Installation only makes the distribution discoverable in the active Python environment. Enable it for the Application:

```bash
reme start plugins='["auto-fin"]'
```

Restart a running service after changing package or enablement state.

## May I edit workspace Markdown directly?

Yes. Files are the source of truth and watchers ingest changes. Keep frontmatter valid, use complete workspace-relative wikilinks, and avoid unconditional concurrent saves.

## May I expose ReMe publicly?

Not with the default configuration alone. Jobs can write and delete, HTTP CORS is permissive, and there is no general authentication layer. Use a controlled network or authenticated TLS reverse proxy and restrict `service.jobs`.

## How should I back up and migrate memory?

Stop writes and back up the complete workspace. `session/`, `resource/`, `daily/`, and `digest/` are the key sources; `metadata/` can be backed up or rebuilt. See [Diagnostics, Backup, and Recovery](./operations.md).

## Why is Studio unavailable?

The base `reme-ai` package has no frontend assets. Install `reme-ai[web]` or `reme-ai[core]`, or set `service.web_static_dir`. Missing Studio assets do not disable the Job API.

## Which capabilities does the running service expose?

```bash
reme help
reme app_config
```

Static documentation describes defaults; plugins and custom configuration may change the active service.
