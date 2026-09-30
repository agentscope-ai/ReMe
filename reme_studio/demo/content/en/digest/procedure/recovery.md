---
name: Protect files and recover derived state
description: Protect user memory before repairing indexes and links.
memory_tags: ["aurora", "local_first", "procedure"]
---

# Recovery procedure

When retrieval results are missing, first check that source files exist, then inspect derived indexes. Do not delete or rewrite user memory to repair an index.

1. Preserve a workspace copy and verify paths and file content.
2. Check that the local ReMe service uses the expected workspace.
3. Inspect ingestion, search indexes, and wikilink graphs according to the issue.
4. `reindex` rebuilds search and tag indexes from ingested chunks. It does not scan the workspace or rebuild the graph; new or changed files still need normal ingestion.
5. Verify results with queries that expose their sources.

Source: [[daily/2026-09-17/validation.md|the validation diary]]. Principle: [[digest/wiki/local-first.md|files are the source of truth]].

This browser demo has no backend index. Its graph and tags are derived directly from current example files. Reset demo restores only this demo copy; it is not a production recovery tool.
