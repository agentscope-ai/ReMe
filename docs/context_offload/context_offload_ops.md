---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Context Offload Ops

## ContextOffloadOp

### Purpose

Manages context window limits by intelligently compacting tool messages and compressing conversation history to reduce token usage while preserving important information.

### Functionality

- Supports three context management modes: `compact`, `compress`, and `auto`
- **Compaction mode**: Stores full content of large tool messages in external files, keeping only previews in context
- **Compression mode**: Uses LLM to generate concise summaries of older message groups
- **Auto mode**: Applies compaction first, then compression if compaction ratio exceeds threshold
- Automatically writes offloaded content to files via `BatchWriteFileOp`
- Preserves recent messages and system messages to maintain conversation coherence
- Configurable token thresholds for both compaction and compression operations

### Parameters

- `messages` (array, **required**):
  - List of conversation messages to process for context offloading
  - Messages are analyzed for token count and processed according to management mode

- `context_manage_mode` (string, optional, default: `"auto"`):
  - Context management mode to use
  - `"compact"`: Only applies compaction to tool messages
  - `"compress"`: Only applies LLM-based compression
  - `"auto"`: Applies compaction first then compression if compaction ratio exceeds threshold
  - Allowed values: `["compact", "compress", "auto"]`

- `max_total_tokens` (integer, optional, default: `20000`):
  - Maximum token count threshold for triggering compression/compaction
  - For compaction: total token count threshold for all messages
  - For compression: excludes `keep_recent_count` messages and system messages
  - Operation is skipped if token count is below this threshold

- `max_tool_message_tokens` (integer, optional, default: `2000`):
  - Maximum token count per individual tool message before compaction is applied
  - Tool messages exceeding this threshold will have full content stored in external files
  - Only a preview is kept in context with a reference to the stored file

- `group_token_threshold` (integer, optional):
  - Maximum token count per compression group when using LLM-based compression
  - If `None` or `0`, all messages are compressed in a single group
  - Messages exceeding this threshold individually will form their own group
  - Only used in `"compress"` or `"auto"` mode

- `keep_recent_count` (integer, optional, default: `1` for compaction, `2` for compression):
  - Number of recent messages to preserve without compression or compaction
  - These messages remain unchanged to maintain conversation context
  - Does not include system messages (which are always preserved)

- `store_dir` (string, optional):
  - Directory path for storing offloaded message content
  - Full tool message content and compressed message groups are saved as files in this directory
  - Required for compaction and compression operations to function properly

- `chat_id` (string, optional):
  - Unique identifier for the chat session
  - Used for file naming when storing compressed message groups
  - If not provided, a UUID will be generated automatically

### Return Value

The operation returns processed messages with reduced token count:
- Compacted tool messages contain previews with file references
- Compressed message groups are replaced with concise summaries in system message
- Metadata includes information about offloaded content and file paths

Example: After processing 50 messages (25,000 tokens) in auto mode, returns 45 messages (15,000 tokens) with 5 tool messages compacted and older conversation history compressed.

