# MCP Service

ReMe exposes its memory operations as MCP (Model Context Protocol) tools, enabling any MCP-compatible agent to read, write, and search long-term memory.

## Overview

The MCP service wraps ReMe Jobs as callable tools over three transports:

- **stdio**: Standard input/output for local agent processes
- **SSE**: Server-Sent Events for network-based agent clients
- **streamable-http**: HTTP streaming for web-based agent frameworks

Each registered non-stream Job becomes an MCP tool with:
- Tool name matching the Job name (e.g., `write`, `read`, `search`, `auto_memory`)
- Tool description from the Job's documented purpose
- Parameters derived from the Job's parameter schema
- Return value as the `answer` field from the Response

## Quick Start

```bash
# Start MCP service over SSE (default)
reme start service.service_type=mcp

# Start MCP service over stdio (for local agents)
reme start service.service_type=mcp mcp.transport=stdio

# Start MCP service over streamable HTTP
reme start service.service_type=mcp mcp.transport=streamable-http
```

The default SSE endpoint is `http://127.0.0.1:2333/sse`.

## Configuration

### Basic Configuration

```bash
reme start \
  service.service_type=mcp \
  service.host=0.0.0.0 \
  service.port=2333
```

### Transport Selection

```yaml
# In custom config.yaml
service:
  service_type: mcp

mcp:
  transport: sse          # sse | stdio | streamable-http
  host: 127.0.0.1
  port: 2333
  tool_error_on_failure: false
```

### Injected Parameters

The MCP server can inject default parameters into every tool call, such as workspace paths or user identifiers:

```yaml
mcp:
  injected_job_kwargs:
    workspace_dir: /path/to/user-workspace
    user_id: agent-001
```

Injected parameters cannot be overridden by the agent client. If a client tries to provide an injected parameter, the server rejects the call with an error.

## Available Tools

### File Operations

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `write` | Create or overwrite a memory file | `path`, `name`, `description`, `content` |
| `read` | Read a file with optional line range | `path`, `start_line`, `end_line` |
| `edit` | Search-and-replace within a file | `path`, `old`, `new` |
| `delete` | Remove a file | `path` |
| `move` | Rename or relocate a file | `src_path`, `dst_path` |
| `list` | Enumerate files under a directory | `path`, `recursive`, `limit` |
| `stat` | Get file metadata | `path` |
| `frontmatter_read` | Read frontmatter only | `path` |
| `frontmatter_update` | Update frontmatter metadata | `path`, `metadata` |

### Search and Retrieval

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `search` | Hybrid BM25 + vector search with link expansion | `query`, `limit`, `min_score`, `vector_weight` |
| `node_search` | Node-level digest search for dream recall | `query`, `limit` |
| `traverse` | BFS over wikilink edges | `path`, `direction`, `depth` |

### Memory Management

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `auto_memory` | Record conversation facts into daily notes | `session_id`, `messages`, `memory_hint` |
| `auto_resource` | Process external resources | `changes` |
| `auto_dream` | Distill daily notes into digest memory | `date` |
| `proactive` | Read daily proactive topics | `date`, `include_content` |
| `daily_list` | List notes under a specific day | `date` |
| `daily_write` | Create a daily note | `date`, `name`, `description`, `content` |

### Indexing

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `reindex` | Rebuild all indexes | - |
| `bm25_search` | BM25-only keyword search | `query`, `limit` |
| `vector_search` | Vector-only semantic search | `query`, `limit` |

## Integration with Agents

### Claude Code

ReMe includes a dedicated Claude Code plugin in `plugins/reme/`. The plugin:
- Bundles the `reme_memory` SKILL.md for memory operations
- Configures MCP connection to the ReMe service
- Provides auto-memory hooks for session recording

See [plugins/reme/README.md](../../plugins/reme/README.md) for setup instructions.

### Codex

A Codex plugin is available in `plugins/codex/` with similar MCP integration.

### Generic MCP Clients

Any MCP-compatible client can connect to ReMe:

```json
// .mcp.json
{
  "mcpServers": {
    "reme": {
      "command": "reme",
      "args": ["start", "service.service_type=mcp", "mcp.transport=stdio"]
    }
  }
}
```

For SSE-based clients:

```json
{
  "mcpServers": {
    "reme": {
      "url": "http://127.0.0.1:2333/sse"
    }
  }
}
```

## Response Format

Each tool call returns a string answer. The answer is designed to be self-contained and actionable for LLM consumers:

- **File results**: Include workspace-relative paths and relevant metadata
- **Search results**: Include file paths, line ranges, scores, and content snippets
- **Error states**: Return clear error messages rather than empty responses
- **Empty results**: Return explicit messages like "No relevant information was found"

Structured metadata continues to be available for programmatic clients via the HTTP service.

## Troubleshooting

### Port Already in Use

```bash
reme start service.port=8181
```

### Missing Dependencies

For MCP to work, ensure `fastmcp` is installed:

```bash
pip install "reme-ai[core]"
```

### Tools Not Visible

Only non-stream Jobs are exposed as MCP tools. Verify the Job is configured with `enable_serve: true` in the configuration.

### Connection Issues

For stdio transport, ensure the `reme` command is on your PATH. For SSE, verify the server is running and the port is accessible:

```bash
curl -s http://127.0.0.1:2333/health_check
```
