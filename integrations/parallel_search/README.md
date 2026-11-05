# Parallel Search MCP example

This opt-in script demonstrates ReMe's `MCPClient` with Parallel's current
streamable-HTTP MCP endpoint. It discovers `web_search` and `web_fetch`, searches
for a topic, then fetches a URL you select. Search and fetch are available
without an API key at lower rate limits. The example prints results only; it
does not call ReMe memory jobs, save files, or ingest web content.
Running it sends the topic and selected URL to Parallel for processing.

## Install and run

From the ReMe repository root, install ReMe and its declared FastMCP dependency:

```bash
python -m pip install -e .
python integrations/parallel_search/example.py "recent ReMe memory updates" https://github.com/agentscope-ai/ReMe
```

Python 3.11 or later is required. For higher provider rate limits, configure an
API key using Parallel's supported MCP authentication; do not commit credentials
to this repository. The script uses a 60-second client timeout and reports
provider rate-limit responses separately.

## Tests

The unit tests mock the MCP client and transport, so they do not contact Parallel:

```bash
python -m pytest tests/unit/test_parallel_search_example.py -q
```

For provider details and current tool arguments, see Parallel's
[MCP quickstart](https://docs.parallel.ai/integrations/mcp/quickstart).
