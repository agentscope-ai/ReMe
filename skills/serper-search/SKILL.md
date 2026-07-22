---
name: serper-search
description: Search the public web through Serper instead of the built-in WebSearch tool. Use when an agent needs current information, general web sources, or URLs not available from local files and project APIs. SERPER_API_KEY is available in the environment.
---

# Serper Search

Use Serper whenever external web search is needed. Prefer local files and project APIs when they already contain the required information.

## Rules

- Do not use the built-in WebSearch tool.
- Do not expose the API key in answers, logs, committed fixtures, or browser/client code.
- Prefer authoritative and primary sources in the returned results.
- Refine the query when freshness, date, company, product, or source type matters.

## API

- Endpoint: `POST https://google.serper.dev/search`
- Auth header: `X-API-KEY: $SERPER_API_KEY`
- Content type: `application/json`
- Required body: `{"q":"query text"}`

## Curl

```bash
curl -s --location "https://google.serper.dev/search" \
  --header "X-API-KEY: $SERPER_API_KEY" \
  --header "Content-Type: application/json" \
  --data '{"q":"apple inc"}'
```

## Response Use

- Prefer `organic` results for normal search answers.
- Use `searchParameters` to confirm the query actually sent.
- Treat `credits` as usage accounting.
- Verify important claims against the linked source rather than treating snippets as complete evidence.
- Summarize results with source titles and links; do not copy large passages from snippets or pages.

## Failure Handling

1. If DNS/network fails, report it as an environment/network issue and retry only when external network is available.
2. If authentication fails, report that `SERPER_API_KEY` is missing or invalid without printing its value.
3. If results are irrelevant, refine the query with names, dates, locations, or source domains.
