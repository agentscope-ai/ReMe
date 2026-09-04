---
title: Diagnostics, Backup, and Recovery
description: ReMe health checks, logs, index maintenance, workspace backup, migration, and recovery.
---

# Diagnostics, Backup, and Recovery

ReMe recovery protects user-owned workspace files and rebuilds catalogs, indexes, and graphs from those sources. Never delete or rewrite user memory merely to repair derived state.

## Quick diagnosis

Run these in order:

```bash
reme find_reme
reme version
reme health_check
reme status
reme app_config
```

- `find_reme` confirms the actual host, port, and PID;
- `version` verifies that the CLI reaches the service;
- `health_check` reports component health;
- `status` estimates stateful component memory and process RSS;
- `app_config` returns the effective configuration with secrets redacted.

## Logs and common symptoms

`log_to_console` and `log_to_file` control logging. For startup failures, inspect the first exception rather than later client connection errors.

| Symptom | Check first |
|---|---|
| CLI cannot find ReMe | `reme find_reme`, process state, startup directory, and port |
| Automatic memory fails | LLM backend, model, API key, and base URL |
| Search is BM25-only | Whether an embedding store is connected to `file_store` |
| New files are absent | Directory, extension, watcher, and `health_check` |
| Studio fails but API works | Installed web extra, static path, and browser console |
| Installed plugin is unavailable | Python interpreter, `plugins` configuration, and service restart |

## Index maintenance

```bash
reme reindex scope=all
reme reindex scope=bm25
reme reindex scope=embedding
```

`reindex` rebuilds BM25 and/or embedding indexes from the current `file_chunks`. It does not scan the workspace, rechunk files, or rebuild the wikilink graph. Diagnose the watcher first when ingestion is the problem.

Rebuild a daily index page separately:

```bash
reme daily_reindex date=2026-09-04
```

## Backup

Stop writes or stop the service, then back up the complete workspace. The most important sources are:

- `session/` for conversation sources;
- `resource/` for external resources;
- `daily/` for daily memory;
- `digest/` for consolidated memory.

`metadata/` contains indexes, graphs, and catalogs. Backing it up accelerates restoration, but it is not the sole source of truth.

Use an explicit, stable absolute `workspace_dir` for durable deployments rather than relying on an incidental `.reme/` under the current directory.

## Migrate a workspace

1. Stop the old service to prevent writes during the copy.
2. Copy the complete workspace while preserving timestamps.
3. Start with the new absolute path:

```bash
reme start workspace_dir=/new/location/reme-memory
```

4. Run `health_check`, `status`, and a representative `search`.
5. Rebuild embeddings if their model or dimensions changed.

Do not push a workspace containing private conversations to a public repository.

## Recover derived state

Do not remove anything until a backup exists. Then:

1. preserve `session/`, `resource/`, `daily/`, and `digest/`;
2. record the effective configuration and component backends;
3. verify that the failure is limited to `metadata/`;
4. move suspect derived state to an isolated backup location;
5. restart with the same configuration and let watchers rebuild;
6. validate search, graph traversal, and daily indexes.

The internal layout of metadata files is not a public automation contract.

## Concurrent editing

When Studio or an editor saves a complete file, pass the mtime from `stat` as `save.expected_mtime`. A save then fails if another actor changed the file after it was opened, avoiding silent overwrites.

File Jobs enforce workspace containment and per-path locking. Do not bypass them to write arbitrary absolute paths.
