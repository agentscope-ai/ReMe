# Auto Fin Plugin

[中文](README_ZH.md)

Auto Fin fetches a rolling window of CLS telegraph news (24 hours by default), selects items related to configured
topics, searches ReMe for useful historical context, and writes one Chinese Markdown report with validated wikilinks.
Current news and topic selection stay in runtime memory; only the final report becomes durable memory. This directory
is an independent Python distribution. Its `reme.plugins` entry point contributes the three Step backends and their Job
configuration; its `reme.configs` entry point exposes the runnable `auto-fin` configuration.

> Auto Fin has no reliable market-price feed. It does not calculate returns, targets, or entry points and is not
> investment advice.

## Quick start

```bash
python -m pip install "reme-ai[core]>=0.4.1.8" reme-auto-fin
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="qwen3.7-plus"
export LLM_BASE_URL="https://your-provider.example/v1"
reme start config=auto-fin job=auto_fin
```

`LLM_MODEL_NAME` defaults to `qwen3.7-plus`. There is no built-in `LLM_BASE_URL`, so set the OpenAI-compatible endpoint
required by the selected provider.

The default topics are `黄金,机器人,半导体`. Override them per run:

```bash
reme start config=auto-fin job=auto_fin topics="黄金,AI,存储芯片"
```

An empty value also uses the defaults.

## Pipeline

```text
CLS public telegraph endpoint (rolling 24 hours)
        ↓
normalize and deduplicate in RuntimeContext
        ↓
topic Agent selects real news IDs in bounded batches
        ↓
research Agent uses memory_search + read on historical memory
        ↓
validate historical wikilinks in code
        ↓
daily/YYYY-MM-DD/auto_fin.md
```

`auto_fin_data_step` signs and paginates the same endpoint used by the CLS website. It starts at the decision time and
stops only after covering the exact preceding 24 hours. Requests are rate-limited and retried; malformed records and
records outside the window are discarded.

`auto_fin_topic_step` receives batches of current news and returns only related `news_id` values. Code ignores unknown
IDs and deduplicates repeated IDs, then preserves the source-news order. If nothing is relevant, the job succeeds as a
skip without writing or sending a report.

`auto_fin_merge_step` receives only selected current news. It exposes `memory_search` and `read`, instructs the Agent to
search no later than yesterday, and keeps current CLS IDs, times, and titles as plain evidence. The prompt limits
wikilinks to historical Markdown actually used by the Agent; the code-level boundary independently keeps only existing,
workspace-relative Markdown targets. Missing, absolute, escaping, backslash, and self-referential targets are degraded
to their readable aliases.

Same-day reruns use the existing report as context and replace it with the revised result. The final write is atomic and
refreshes the daily index. No JSONL, intermediate Markdown, or structured Agent output is written.

## Parameters

| Parameter          |                Default | Purpose                                                                  |
|--------------------|-----------------------:|--------------------------------------------------------------------------|
| `date`             |                   `""` | Empty uses today in Shanghai; an explicit value must equal today         |
| `now`              |                   `""` | Optional ISO 8601 decision time for testing or replay                    |
| `topics`           | `"黄金,机器人,半导体"` | Comma-separated topics; empty also uses these defaults                   |
| `window_hours`     |                   `24` | Rolling number of hours of CLS telegraph news to fetch; must be positive |
| `request_interval` |                   `10` | Minimum delay in seconds after every CLS request attempt; may be zero    |
| `max_retries`      |                    `3` | Maximum attempts for each CLS page request; must be at least one         |

The plugin-provided schedules run daily at 09:30, 11:30, and 18:00 in `Asia/Shanghai`.

## Output

```text
.reme/daily/YYYY-MM-DD/auto_fin.md
```

The report includes a title, description, current CLS evidence, historical analysis, contextual wikilinks, and a fixed
non-investment disclaimer. Network errors and invalid Agent output fail explicitly; no relevant current news is a
successful skip.

## Validation

```bash
python -m pytest plugin/auto-fin -v
```

Unit tests mock the CLS and Agent boundaries and do not contact external services.
