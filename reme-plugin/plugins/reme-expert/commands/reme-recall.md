---
description: Deep recall — graph-aware hybrid retrieval on a query. Use when SessionStart-style auto-injection isn't enough.
argument-hint: <query, optionally including [[Anchor]]>
---

Call the `mcp__reme__memory_graph_search` MCP tool with:

```
query: $ARGUMENTS
max_results: 8
graph_depth: 1
```

If the query contains a `[[wikilink]]`, the search will seed BFS at that file. If you want to seed at a specific topic explicitly, edit the call to add `seeds: [<absolute-path>]`.

After the call returns, present the top hits to the user: for each, the file path, the `graph_hop` distance from the seed, and a 1-line excerpt. Don't paraphrase the chunks — they may be load-bearing for the user's question.
