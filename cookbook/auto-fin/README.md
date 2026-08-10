# Auto Fin Cookbook

[中文](README_ZH.md)

Auto Fin is a local-first, file-native topic-news research workflow. It reads a free local CLS telegraph JSONL file,
interprets current news around user-supplied topics, and lets the final agent search and read earlier ReMe news and
analysis notes. The resulting report links back to current and historical evidence through validated,
workspace-relative wikilinks.

> Auto Fin has no reliable market-price feed. It does not calculate returns, targets, or entry points and is not
> investment advice.

## Quick start

Prepare a local CLS telegraph JSONL file first. Each line must be a JSON object containing `id`, `ctime` (a Unix
timestamp), `title`, and `content`:

```json
{"id": 2448247, "ctime": 1786323600, "title": "Example", "content": "News text"}
```

Keep the feed outside version control. Auto Fin only reads this file and does not fetch news from the network.

Then run topic research:

```bash
python -m pip install -e ".[core]"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=auto_fin topics="gold,robotics,semiconductors"
```

The default input is `datasets/cls_news_last_7_days.jsonl`. Override it with the `news_file` parameter or environment:

```bash
export AUTO_FIN_NEWS_FILE=/data/cls_news.jsonl
```

## Pipeline

```text
free CLS JSONL
      ↓
daily/YYYY-MM-DD/auto_fin_news.md
      ↓
update the ReMe index
      ↓
research agent (memory_search + read)
      ↓
write contextual wikilinks in prose; validate paths in code
      ↓
daily/YYYY-MM-DD/auto_fin.md
```

`auto_fin_data_step` only reads local files. It splits JSONL into daily Markdown in `Asia/Shanghai`; today's note stops
at the decision time. The default lookback is seven calendar days. Existing prior-day notes are reused and today's note
is refreshed.

`auto_fin_merge_step` is the only model-facing business step. It receives current news, topics, the latest prior-day
report, and any existing same-day report. It exposes only two ReMe job tools:

- `memory_search` recalls earlier news and `auto_fin.md` reports;
- `read` opens likely matches as complete Markdown.

The agent searches by topic and important event, internally classifies results as similar, related, or unrelated, and
only cites documents it actually read and used. Same-day reruns refine today's report. Historical reports remain
immutable records of earlier judgments.

## Structured output

The final agent returns one Markdown-shaped contract with three required fields:

```json
{
  "title": "Gold and Semiconductor News Review",
  "description": "Tracks safe-haven demand and memory-chip supply changes.",
  "body": "## Current assessment\n\n..."
}
```

There is no duplicate `sources` or citation-reason list. Code normalizes Markdown heading markers in the title, writes
the final file, appends the fixed disclaimer, and refreshes the index. The validated structured result is retained at
`resource/YYYY-MM-DD/auto_fin_merge_output.json`.

## Wikilink contract

The final structured output contains only `title`, `description`, and `body`. Historical wikilinks appear once, directly
inside the relevant prose in `body`; there is no duplicate `sources` output. Code keeps only existing workspace-relative
Markdown targets, rejects absolute, escaping, backslash, and self-referential paths, and preserves valid links in place.

Invalid links degrade to their readable alias without deleting the surrounding analysis. Links must follow Dream
Integrate's convention and appear in a sentence that explains the relationship. If the body does not cite today's news
file, code appends a deterministic `## 来源` link. The day index is refreshed after writing the report.

## Parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `date` | `""` | Empty uses today in Shanghai; an explicit date must equal today |
| `now` | `""` | ISO 8601 simulated current time for testing or replay |
| `topics` | `""` | Comma-separated research topics; empty means unrestricted |
| `news_file` | `""` | Local CLS JSONL override; empty uses step configuration or the environment |

The runtime `news_file` parameter wins over step configuration. The built-in configuration reads
`AUTO_FIN_NEWS_FILE`, falling back to `datasets/cls_news_last_7_days.jsonl`. `topics` constrains semantic relevance;
a keyword mention without a real topic relationship should be ignored by the agent.

The built-in schedules remain 09:30, 11:30, and 18:00. Without a trading-calendar dependency, weekend and holiday news
can also produce reports.

## Outputs

```text
reme_workspace/
├── daily/YYYY-MM-DD/
│   ├── auto_fin_news.md
│   └── auto_fin.md
└── resource/YYYY-MM-DD/
    └── auto_fin_merge_output.json
```

The JSONL and daily Markdown are source material; `auto_fin.md` is a reviewable research judgment; indexes and graph
state remain rebuildable from those files.

A missing or malformed JSONL file, failed agent call, or invalid model structure stops the job explicitly. A missing or
unsafe historical wikilink does not fail report generation; it degrades to readable plain text. Code always ensures that
the current-news source link is present.

## Validation

```bash
pytest tests/unit/test_auto_fin.py -v
```

Unit tests use local JSONL and a mock agent and do not contact news, model, or other external services.
