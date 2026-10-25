# Procedure Memory

An optional ReMe plugin that turns a local trajectory file into reusable procedural experiences. It keeps the original bytes, stores each experience as one JSON file, and provides two hybrid vector + BM25 search views. Existing factual memory is unchanged.

## Install and configure

From the ReMe repository:

```bash
pip install -e '.[core,qwen-tokenizer]' -e plugins/procedure-memory
```

Enable with `plugins='["procedure-memory"]'`. Use the usual `LLM_*` settings for extraction and `EMBEDDING_BACKEND`, `EMBEDDING_MODEL_NAME`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` for vectors. The plugin explicitly enables embeddings for its own stores. Set embedding dimensions through `components.as_embedding.procedure.dimensions` when your provider needs a different value.

`PROCEDURE_TOKENIZER_SOURCE` selects a Hugging Face tokenizer name or local tokenizer file/directory. The default is `Qwen/Qwen3.8-27B`. Counts must be exact for the configured tokenizer; unavailable tokenizers cause a resumable failure instead of an estimated limit. Select the inference model's tokenizer for model-specific counts. The tokenizer may download its assets on first use; a local tokenizer path supports offline operation.

## Local extraction (one-shot CLI)

Import is private to the local CLI or in-process `Application`; it is not an HTTP/MCP upload endpoint. One file is one trajectory, including a JSONL file with many records. Stop any other ReMe process using the same workspace before running a one-shot import or rebuild.

```bash
reme start job=procedure_auto_memory plugins='["procedure-memory"]' \
  file=/absolute/path/run.jsonl filename=run.jsonl request_id=run-001
```

`file` and `filename` are required. `memory_hint` optionally focuses extraction; `source_id` is an optional caller label. A stable `request_id` reuses the copied path and processing checkpoint. Resume with the same original `file` path and `filename`; once the workspace copy is published, the original may be deleted or moved. The saved copy is verified against its recorded checksum. If the original still exists, changed bytes are rejected. Without a saved copy, the original is required. Without a request ID, every invocation is a new import; the response returns a generated request ID for subsequent resume. Reusing an ID with a different path/name or incompatible processing settings fails explicitly. Context/reserve and SDK iteration budgets may be adjusted to resume unfinished work without changing segmentation.

Supported reading formats: plain text (including extensionless files), Markdown, JSON, JSONL and logs. JSON/JSONL retain all fields. Malformed or unsupported input produces an incomplete response and retains the saved original. Imports preserve bytes and extensions; they are never rewritten as `.txt`. Files are limited to 20 MiB by default (`components.procedure_memory.default.max_file_bytes`).

Long trajectories are processed segment by segment with a rolling summary. The extraction agent can search, read, create, revise or skip experiences; there is no experience-count limit. Each write validates fields and sources and immediately updates both indexes. Later segments may rewrite all three text fields. Token errors ask the agent to shorten its fields, without silently truncating experiences. Summary repairs have a bounded retry budget.

## Service search

After local imports finish, use the normal two-stage service flow:

```bash
# Terminal 1
reme start plugins='["procedure-memory"]'

# Terminal 2
reme procedure_search query='recover after an index configuration change' index=when_to_use
reme procedure_search query='inspect logs before rebuilding' index=when_to_use_content limit=5
reme procedure_read path=procedures/rebuild-search-after-config-change.json
```

`when_to_use` indexes only `when-to-use`; `when_to_use_content` indexes `when-to-use + "\n\n" + content`. Each experience is exactly one record in each view, without chunks. Both views use separate BM25 indexes and graphs and a shared embedding cache. Search combines vector and BM25 rankings with weighted RRF (default vector weight 0.7), and returns full current experiences, their workspace-relative paths, versions and scores. Repeated searches may return the same experience.

For auto-memory integration, the agent receives `procedure_search`, `procedure_read`, `procedure_create`, `procedure_update` as Job tools. Runtime-only write capabilities and original-source paths are injected by the application. Create/update cannot be used outside an active extraction run. Updates require `expected_version` from a search/read and preserve existing sources. Source excerpts are available via `procedure_read kind=trajectory path=session/trajectories/run.jsonl offset=0 length=8192`; offsets and lengths count decoded characters, with an additional token cap.

## User files and derived state

```text
<workspace>/
  session/trajectories/run.jsonl
  session/trajectories/run-2.jsonl
  procedures/rebuild-search-after-config-change.json
  metadata/procedure/default/{imports,operations,processing}/
  metadata/{file_store,file_graph,keyword_index,embedding_store}/...
```

`session` follows ReMe's `session_dir`. Configure `procedure_dir` and `trajectory_subdir` under `components.procedure_memory.default`. All destinations and tool reads stay inside the workspace and their configured directories. Only explicit import reads can use a source path outside the workspace.

An experience has exactly these four fields, without an ID:

```json
{
  "description": "Rebuild search after a configuration change",
  "when-to-use": "After changing search configuration when results appear stale",
  "content": "Rebuild the affected indexes, then verify a known query before resuming.",
  "sources": ["session/trajectories/run.jsonl"]
}
```

The agent picks a meaningful lowercase-hyphenated JSON filename; the path is its stable identity. An existing experience name requires an update or another meaningful name. Original trajectory name collisions instead get `-2`, `-3`, etc. before the last extension. Pending import reservations also retain their names. New imports are not deduplicated by content. `sources` contain only actual copied trajectory paths, deduplicated by the runtime. There is no `source.json` or public `trace_id`.

Defaults in [plugin.yaml](src/reme_procedure_memory/plugin.yaml): segment 98,304 tokens (96K); rolling summary target 8,192 (8K); description 128; when-to-use 512; content 2,048. Summary and experience-field validation internally allow 10% slack: default acceptance ceilings are 9,011 tokens for summary, 140 for description, 563 for when-to-use and 2,252 for content. The agent receives only the configured targets, without the slack. Segmentation and context budgets remain strict. The initial input, summary and tool schemas must fit a 262,144-token context with a 4,096-token reserve; configure a model with sufficient context capacity. Adjust these under `components.procedure_memory.default.token_limits`; field limits apply to creation and updates. SDK iteration limits are technical execution budgets: unfinished processing remains resumable, rather than imposing a count limit on experiences. Existing checkpoints require their original segmentation settings when resuming.

## Recovery and maintenance

The operation journal records the destination, target content and stable scoped `operation_key` before publishing an experience. If indexes fail after a file commit, the write tool returns a failed JSON response with `status=index_pending`, the saved path and version. Retry the identical operation for temporary failures. For a permanently rejected payload, use `procedure_read`, then `procedure_update` on the same path with its current `expected_version` and a new `operation_key`; do not create a replacement filename. The durable correction supersedes the old failed intent, including across interrupted corrections and restarts.

Failed files remain intact and are temporarily excluded from **both** search views. Startup and unrelated trajectories continue. Search returns the available experiences with `index_complete=false` and a `pending_indexes` list containing the excluded paths, versions and errors, rather than silently returning stale records. A corrected file re-enters both views after synchronization. A segment with its own pending writes cannot advance; resume lets the agent run again to correct them. Structural index failures that prevent safe exclusion still fail explicitly.

Checkpoints save segment position, bounded summary, processing fingerprint, recoverable input and final reply. A resumed segment receives its prior operation ledger so it can read completed experiences and continue; it does not assume the model will repeat identical decisions. A saved final reply is reused if the segment checkpoint failed to advance. Restart reconciles pending writes and rebuilds both views from experience JSON. Rebuilds never delete or rewrite experience or trajectory files.

For explicit maintenance, with other workspace writers stopped:

```bash
reme start job=procedure_reindex plugins='["procedure-memory"]'
```

Reindex reports invalid JSON files separately. This version supports one process per workspace, serializes file extraction within that process, and does not watch manual edits. Trajectories and summaries are not indexed. Automatic deletion, renaming, merging and splitting of experiences are not supported.

## Validation

```bash
cd plugins/procedure-memory
pytest -q
```

The tests use isolated temporary workspaces, real ReMe Jobs/FileStores/BM25, and fake agents/embeddings/token counters. No model credentials or network are needed. The public schema is in [schema.py](src/reme_procedure_memory/schema.py); storage and recovery contracts are in [store.py](src/reme_procedure_memory/store.py).
