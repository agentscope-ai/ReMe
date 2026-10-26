# MemLens plugin

Install with `pip install -e plugins/memlens` and load the `memlens-benchmark` configuration
with the `memlens` plugin. The executable ReMe Image evaluation lives
in [`benchmark/memlens`](../../benchmark/memlens).

`auto_memory` extends native Auto Memory with a MemLens-specific system prompt.
It retains native source-session persistence and interleaved image inputs; `include_images`
selects whether the memory model sees historical images. No question or reference answer
is given during ingestion.

`agentic_answer` accepts the official answer prompt as `query`, optional `query_time`,
and native image DataBlocks as `query_images`. It creates a fresh AgentScope Agent for
each question with BM25 `search` and scoped `read`. Question images remain visible in
both the text-memory and image-memory settings.
Its `max_iters` input defaults to 30 ReAct iterations and applies only
to the current answer Agent. The effective value is returned in answer metadata;
Auto Memory and context compression keep their existing settings.

`include_source_sessions` defaults to `true`: memory prompts include exact physical
JSONL line references, and the answering Agent can search/read those source sessions.
When false, only generated daily/digest markdown is searchable and readable, regardless
of whether an old session index exists. Link expansion is disabled. JSONL persistence
is unchanged. The runner must also omit session directories from index updates for
the source-disabled setting.

With `include_images=true`, native Auto Memory saves input images beneath
`session/images/` and supplies their paths to the memory Agent. The answer tool
`read_image(path)` calls ReMe's existing `read_image` Job and converts its successful
image response to an AgentScope DataBlock. The plugin supplies the native Job with
`max_bytes=52428800` (50 MiB). Image inputs retain their original bytes and resolution.
Native size errors and non-image results
stay textual tool errors; the generic ReMe Job is unchanged.

Image access is independent of `include_source_sessions`: a source-disabled Agent
can still reopen images cited by its memory notes. ReMe Text has no historical-image
tool. All image paths are confined to the current scene's session image directory.
The runner retains official Image IDs as text beside the corresponding image blocks;
the IDs are answer evidence labels, not a separate registry or tool argument.
Do not put answer keys, judge results, another scene's data, or agent state under the
daily/digest/source evidence directories. Every scene starts from a separate workspace;
sessions within one scene accumulate in order. Query state and search deduplication
are independent even when questions execute concurrently.

The plugin preserves the ReMe Image experiment's thinking, output budgets and context
settings. The runner selects the candidate model; scoring belongs to `memlens-judge`.
Answer metadata includes portable usage and the raw final-message usage when available;
missing provider cache/reasoning counters are not reported as zero.
