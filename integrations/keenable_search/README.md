# Keenable Search MCP example

This opt-in script demonstrates ReMe's `MCPClient` with Keenable's public
streamable-HTTP MCP endpoint. It discovers `search_web_pages` and
`fetch_page_content`, searches for a topic, then fetches a URL you select.
Both tools work without an account or API key. The example prints results only;
it does not call ReMe memory jobs, save files, or ingest web content.
Running it sends the topic and selected URL to Keenable for processing.

## Install and run

From the ReMe repository root, install ReMe and its declared FastMCP dependency:

```bash
python -m pip install -e .
python integrations/keenable_search/example.py "recent ReMe memory updates" https://github.com/agentscope-ai/ReMe
```

Python 3.11 or later is required. The example connects anonymously by default.
For higher rate limits, set `KEENABLE_API_KEY`; it is sent as the `X-API-Key`
header and is not persisted. The script uses a 60-second client timeout and
reports provider rate-limit responses separately.

## Tests

The unit tests use mocked HTTP responses through ReMe's real MCP client, so they do not contact Keenable:

```bash
python -m pytest tests/unit/test_keenable_search_example.py -q
```

For provider details and current tool arguments, see Keenable's
[MCP server docs](https://docs.keenable.ai/mcp-server).
