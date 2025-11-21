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

# Context Reload Ops

## GrepOp

### Purpose

Provides a powerful text search capability for locating offloaded content by pattern matching. This operation enables efficient content-based search using regular expressions, making it easy to find specific tool messages or compressed groups in the context store.

### Functionality

- Supports full regular expression syntax (e.g., `"log.*Error"`, `"function\\s+\\w+"`)
- Search across multiple files in specified directories
- Filter search targets using glob patterns (e.g., `"*.txt"`, `"*.json"`)
- Limit result count to avoid overwhelming output
- Ideal for locating offloaded tool messages and compressed message groups
- Returns matching lines with file paths and line numbers

### Parameters

- `pattern` (string, **required**):
  - The regular expression pattern to search for in file contents
  - Supports full regex syntax for complex pattern matching
  - Examples: `"stored in"`, `"tool_call_.*\\.txt"`, `"compressed_group_.*_\\d+"`

- `path` (string, optional, default: current working directory):
  - The directory to search in
  - Typically set to the context store directory (e.g., `/workspace/context_store`)
  - Searches recursively through all subdirectories

- `glob` (string, optional):
  - Glob pattern to filter which files to search
  - Improves performance by limiting search scope
  - Examples: `"*.txt"` (only text files), `"*.json"` (only JSON files), `"compressed_*.json"`

- `limit` (number, optional):
  - Maximum number of matching lines to return
  - Shows all matches if not specified
  - Useful for large result sets to avoid token overflow
  - Example: `50` returns at most 50 matching lines

### Return Value

The operation returns search results with matching lines:
- List of matches including file path, line number, and content
- Total count of matches found
- Each match provides full context for the matched line

Example: Searching for `"stored in /.*\\.txt"` in `/workspace/context_store` returns 12 matches across 8 files, each showing the file path and the line containing the storage reference.

## ReadFileOp

### Purpose

Reads and returns the complete content of offloaded files, enabling on-demand access to compacted tool messages and compressed conversation history. Supports pagination for efficient handling of large files.

### Functionality

- Reads complete file content from specified absolute path
- Supports pagination with offset and limit for large files
- Works with both text and JSON files
- Essential for retrieving full content of compacted tool messages
- Enables access to original message groups before compression
- Returns file metadata including size and line count

### Parameters

- `absolute_path` (string, **required**):
  - The absolute path to the file to read
  - Must be a complete path; relative paths are not supported
  - Examples:
    - `/workspace/context_store/tool_call_123.txt`
    - `/workspace/context_store/compressed_group_research_session_001_0.json`

- `offset` (number, optional):
  - For text files, the 0-based line number to start reading from
  - Used in combination with `limit` for pagination
  - Useful for reading large files in chunks
  - Example: `0` starts from the first line, `100` starts from line 100

- `limit` (number, optional):
  - For text files, maximum number of lines to read
  - Used with `offset` to implement pagination
  - If omitted when `offset` is provided, reads from offset to end of file
  - Example: `100` reads up to 100 lines

### Return Value

The operation returns file content with metadata:
- Full content of the file (or specified line range)
- File path and size information
- Line count for text files
- Content is returned as a string (text files) or can be parsed as JSON

Example: Reading `/workspace/context_store/tool_call_123.txt` returns the complete 8,200-token tool output that was originally compacted, along with file size (8,200 bytes) and line count (450 lines).

## Usage Pattern: Combining Grep and ReadFile

These two operations work together to implement context reload functionality:

**Step 1: Use Grep to Locate Files**
```python
# Find all tool messages
grep(pattern="tool_call_", path="/workspace/context_store", glob="*.txt")
# Returns list of matching files
```

**Step 2: Use ReadFile to Retrieve Content**
```python
# Read specific file found by grep
read_file(absolute_path="/workspace/context_store/tool_call_123.txt")
# Returns full content
```

**Common Reload Scenarios:**

1. **Reload Specific Tool Message**
   - Grep: Search for file reference in message history
   - ReadFile: Load full content from identified file

2. **Reload Chat Session**
   - Grep: Find all files for specific chat_id
   - ReadFile: Load each file to reconstruct complete history

3. **Preview Then Load**
   - ReadFile (offset=0, limit=50): Preview first 50 lines
   - ReadFile (no limit): Load complete file if preview looks relevant

4. **Batch Reload**
   - Grep: Find multiple files matching pattern
   - ReadFile: Iterate through results to load all content
