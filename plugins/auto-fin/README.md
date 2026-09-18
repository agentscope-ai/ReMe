# Auto Fin Plugin

[中文](README_ZH.md)

Auto Fin fetches a rolling window of CLS telegraph news (24 hours by default), groups items by configured
topics, researches each topic against ReMe history, and writes one Chinese Markdown report with validated wikilinks.
Current news and topic selection stay in runtime memory; only the final report becomes durable memory. This directory
is an independent Python distribution. Its single `reme.plugins` entry point exposes a `plugin.yaml` containing the
three Step backends and their Job configuration under `application_defaults`. Enable the installed plugin explicitly
through `plugins=["auto-fin"]`.

> Auto Fin has no reliable market-price feed. It does not calculate returns, targets, or entry points and is not
> investment advice.

## Quick start

### 1. Install ReMe and Auto Fin

```bash
python -m pip install "reme-ai[core]>=0.4.1.12"
reme plugins install plugins/auto-fin
```

### 2. Configure the model environment

Configure the LLM environment variables as described in the
[ReMe model-configuration guide](../../README.md#optional-model-configuration). Other compatible models and providers
can also be used.

### 3. Start ReMe with the plugin

```bash
reme start plugins='["auto-fin"]'
```

With no explicit `config`, ReMe loads `default.yaml` and adds the plugin to that service.

From another terminal, call the running HTTP service through ReMe's CLI client:

```bash
reme auto_fin topics="黄金,AI,存储芯片"
```

Or call its HTTP endpoint directly:

```bash
curl -s http://127.0.0.1:2333/auto_fin \
  -H 'Content-Type: application/json' \
  -d '{"topics":"黄金,AI,存储芯片"}'
```

The HTTP service also exposes the same Job as the `auto_fin` MCP tool at `/mcp`. The default topics are
`黄金,机器人,半导体`; an empty value also uses these defaults.

To host the application with both JSON and MCP access:

```bash
reme start plugins='["auto-fin"]' \
  service.backend=http
```

Custom application configs must provide `agent_wrapper.default`, a `file_store.default` with an enabled tag index, and
the `search`, `list_tags`, `frontmatter_read`, and `frontmatter_update` Jobs used by Auto Fin and automatic
tagging.

## Pipeline

```text
CLS public telegraph endpoint (rolling 24 hours)
        ↓
normalize and deduplicate in RuntimeContext
        ↓
topic Agent returns topic-to-news-ID mappings in prompt-sized batches
        ↓
one research Agent per topic examines its latest 20 articles and searches history up to three times
        ↓
combine topic results and validate historical wikilinks in code
        ↓
daily/YYYY-MM-DD/auto_fin.md
        ↓
generate memory tags; the background file watcher refreshes indexes
```

`auto_fin_data_step` signs and paginates the same endpoint used by the CLS website. It starts at the decision time and
stops only after covering the exact preceding 24 hours. Requests are rate-limited and retried; malformed records and
records outside the window are discarded.

`auto_fin_topic_step` batches current news under a 100,000-character full-prompt limit and returns related `news_id`
values for each topic. Code ignores unknown IDs and deduplicates repeated IDs, then preserves source-news order. One article may belong to multiple topics. If nothing is relevant, the job succeeds as a
skip without writing a report.

`auto_fin_merge_step` researches each nonempty topic with its latest 20 articles. It exposes only `search`, enforces a three-call search budget per topic, and keeps current CLS IDs,
times, and titles as plain evidence. The prompt limits
wikilinks to historical Markdown actually used by the Agent; the code-level boundary independently keeps only existing,
workspace-relative Markdown targets. Missing, absolute, escaping, backslash, and self-referential targets are degraded
to their readable aliases.

Same-day reruns use up to the first 30,000 characters of the existing report as context and replace it with the revised result. The combined final write is atomic and
refreshes the daily index. The workflow then runs `auto_tag_step` to update the generated report's memory-tag
frontmatter; the normal background file watcher observes that source-file change and refreshes derived indexes. No
JSONL, intermediate Markdown, or structured Agent output is written.

## Parameters

| Parameter          |                Default | Purpose                                                                  |
|--------------------|-----------------------:|--------------------------------------------------------------------------|
| `date`             |                   `""` | Empty uses today in Shanghai; an explicit value must equal today         |
| `now`              |                   `""` | Optional ISO 8601 decision time for testing or replay                    |
| `topics`           | `"黄金,机器人,半导体"` | Comma-separated topics; empty also uses these defaults                   |
| `window_hours`     |                   `24` | Rolling number of hours of CLS telegraph news to fetch; must be positive |
| `request_interval` |                   `10` | Minimum delay in seconds after every CLS request attempt; may be zero    |
| `max_retries`      |                    `3` | Maximum attempts for each CLS page request; must be at least one         |

The plugin cron Job starts with the application and runs daily at 09:00 in the application timezone, which defaults to `Asia/Shanghai`. The rolling window uses timestamps and may cross calendar days. Report completion depends on news volume and model latency.

## Output

```text
.reme/daily/YYYY-MM-DD/auto_fin.md
```

The report includes a title, description, current CLS evidence, historical analysis, contextual wikilinks, and a fixed
non-investment disclaimer. Network errors and invalid Agent output fail explicitly; no relevant current news is a
successful skip.

## Validation

```bash
python -m pytest plugins/auto-fin -v
```

Unit tests mock the CLS and Agent boundaries and do not contact external services.
