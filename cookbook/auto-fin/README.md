# Auto Fin Cookbook

[中文](README_ZH.md)

Auto Fin is a local-first, file-native ETF event-research workflow. It collects CLS news and market data through
Tushare, identifies current news related to a configured ETF list, retrieves comparable events from local ReMe memory,
calculates observed post-event returns, and writes a Chinese research report.

> Auto Fin is for event research and holding-period reference only. It is not investment advice, does not connect to a
> broker, and does not place or simulate trades.

The workflow is assembled by
[`daily_cookbook.yaml`](../../reme/config/daily_cookbook.yaml). Its public schemas are in
[`reme/schema/auto_fin.py`](../../reme/schema/auto_fin.py), and its four steps are in
[`reme/steps/cookbook/auto_fin/`](../../reme/steps/cookbook/auto_fin/).

## Quick start

Auto Fin requires Python 3.11 or newer, the `core` dependencies, a Tushare token, and an available AgentScope LLM.

```bash
python -m pip install -e ".[core]"
export TUSHARE_TOKEN="your-tushare-token"
export LLM_API_KEY="your-api-key"
reme start config=daily_cookbook job=auto_fin
```

The built-in LLM component defaults to `qwen3.7-plus`. `LLM_BASE_URL` has no built-in value, so set it when your
provider requires a custom OpenAI-compatible endpoint. The model and endpoint can be overridden with
`LLM_MODEL_NAME` and `LLM_BASE_URL`.

The default workspace is `reme_workspace/` beneath the process working directory. Override it with
`DAILY_PAPER_WORKSPACE_DIR`; Auto Fin and Daily Paper share this setting.

Dates and times use `Asia/Shanghai`. An explicit `date` must be today's date:

```bash
reme start config=daily_cookbook job=auto_fin date=2026-08-07
```

Auto Fin checks the SSE trading calendar first and skips the whole workflow on a closed market day.

## Pipeline

```text
Tushare trade calendar
        │
        ├─ closed day ──► skip
        ▼
Collect CLS news + configured ETF history
        ▼
Update the ReMe index
        ▼
Select ETF/news relationships with an agent
        ▼
Search local memory for comparable historical news
        ▼
Select same/opposite events with an agent + calculate D1/D2/D3/D5 returns in code
        ▼
Generate report with an agent ──► refresh day index ──► DingTalk (optional)
```

| Step | Responsibility | Agent |
|---|---|---|
| `auto_fin_data_step` | Check the trading day, maintain news, and cache configured ETF market history | No |
| `auto_fin_topic_step` | Select direct relationships between today's news and configured ETFs | Yes |
| `auto_fin_history_step` | Retrieve comparable news, validate selections, and calculate observed returns | Yes |
| `auto_fin_merge_step` | Combine prepared evidence and the previous report into the final Markdown | Yes |

All three model-facing steps use structured Pydantic output. Agents make semantic judgments; code owns identifier
validation, source resolution, market calculations, and file writes.

## Data and selection boundaries

### News

`auto_fin_data_step` calls Tushare `major_news` with `src="财联社"`. The default lookback is 60 calendar days including
today. Existing files for earlier days are reused, while today's file is always overwritten with news from 00:00
through the current decision time. Large responses are recursively split when a request returns at least 400 rows.

Each item is stored in `daily/YYYY-MM-DD/auto_fin_news.md` with a stable ID made from its publication timestamp and a
short content hash. The current-event set used by the Topic step is today's complete file, not an increment since an
earlier run.

### Configured ETFs

The built-in configuration currently enables:

- `518880.SH`
- `159530.SZ`
- `512760.SH`

Other examples remain commented out in `daily_cookbook.yaml`. For each enabled code, the Data step resolves its name
through `etf_basic`, then pages backward through `fund_daily` and `fund_adj` and rewrites its complete local JSONL
history. A missing ETF name fails the run.

The Topic agent receives only the configured ETF code/name pairs and today's locally stored news. It may retain up to
`current_news_limit_per_etf` valid, unique news references per ETF (10 by default). Unknown ETF codes, unknown news IDs,
empty reasons, duplicates, and ETFs with no accepted event are removed by code.

## Historical comparison and returns

For every accepted current ETF/news pair, `auto_fin_history_step` calls the configured `memory_search` job over the
60-day news window, ending yesterday. `historical_search_limit` controls the maximum search results requested per
current event. Only search hits whose path is named `auto_fin_news.md` contribute candidate IDs; the step rereads the
source Markdown and resolves those IDs before calling the History agent.

The History agent may select at most five candidates by default and labels each relationship `same` or `opposite`.
Code discards unknown or duplicate IDs and empty reasons, then calculates adjusted cumulative returns for D1, D2, D3,
and D5:

- For an event before 15:00 on a trading day, the adjusted same-day close is the entry; D1 is the next trading close.
- For an event at or after 15:00, the adjusted next-trading-day open is the entry; D1 is that day's close.
- If an entry or horizon cannot be calculated from valid positive prices and adjustment factors, that value is `null`.

The final agent receives the fixed ETF list, all current and historical evidence, `same`/`opposite` directions, computed
returns, and the most recent earlier `auto_fin.md`. It decides whether the evidence supports a recommendation or an
explicit wait-and-see conclusion; the code does not calculate a score, expected return, or mandatory holding period.

## Outputs

```text
reme_workspace/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── auto_fin_news.md
│       └── auto_fin.md
└── resource/
    ├── fin/
    │   ├── etfs.json
    │   ├── 518880.SH.jsonl
    │   └── <other-configured-ETF>.jsonl
    └── YYYY-MM-DD/
        ├── auto_fin_topic_output.json
        ├── auto_fin_history_001_output.json
        ├── ...
        ├── auto_fin_analysis.jsonl
        └── auto_fin_merge_output.json
```

The daily news and report are user-owned Markdown. `resource/fin/` contains the market cache used for deterministic
return calculations. Date-scoped JSON/JSONL files preserve structured agent replies and prepared analyses. Writes use
same-directory temporary files and atomic replacement; the day index is refreshed after the report is written.

## Parameters and defaults

Public job parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `date` | `""` | Empty uses today in `Asia/Shanghai`; a value must be strict `YYYY-MM-DD` and equal today |
| `historical_search_limit` | `10` | Maximum `memory_search` results requested for each current event; minimum 1 |

Relevant job settings in `daily_cookbook.yaml`:

| Setting | Default | Purpose |
|---|---:|---|
| `etf_codes` | three enabled codes above | Fixed ETF research universe |
| `news_lookback_days` | `60` | Local news and historical-search window |
| `current_news_limit_per_etf` | `10` | Maximum accepted current events per ETF |
| `historical_news_limit` | `5` | Maximum comparable events retained per current event |

There is no public `force` parameter. Earlier news files are reused, today's news and all configured ETF market files
are refreshed, and same-day report/resource paths are overwritten on each successful run.

## Environment and scheduling

| Variable | Required | Purpose |
|---|---|---|
| `TUSHARE_TOKEN` | Yes | Trading calendar, CLS news, ETF metadata, prices, and adjustment factors |
| `LLM_API_KEY` | Provider-dependent | Shared AgentScope LLM credentials; config defaults to an empty value |
| `LLM_MODEL_NAME` | No | Defaults to `qwen3.7-plus` |
| `LLM_BASE_URL` | Provider-dependent | OpenAI-compatible endpoint; no built-in default |
| `TUSHARE_MIRROR_URL` | No | Replaces the Tushare SDK HTTP URL after trimming a trailing slash |
| `DAILY_PAPER_WORKSPACE_DIR` | No | Shared standalone cookbook workspace |
| `DINGTALK_*` | No | Optional DingTalk application, robot, and group settings |

The optional mirror can be configured, for example, as:

```bash
export TUSHARE_MIRROR_URL="http://112.124.63.173:4000/tushare"
```

`auto_fin_1200_cron` runs every day at 12:00 in `Asia/Shanghai`. The cron fires on weekends and holidays, but the Data
step then skips the remaining workflow when Tushare reports that the date is not an SSE trading day.

To send a completed report, configure `DINGTALK_APP_KEY`, `DINGTALK_APP_SECRET`, `DINGTALK_ROBOT_CODE`, and the
comma-separated `DINGTALK_CONVERSATION_IDS`. With no conversation IDs, delivery is a no-op.

## Agent and failure boundaries

Auto Fin and Daily Paper share the tool-free `default` AgentScope wrapper. Built-in and configured job tools are not
exposed to their model calls. Auto Fin itself invokes `memory_search` in deterministic step code; this is not an agent
tool call. The separate interactive `dingtalk_wait` step has its own `bash` and ReMe job-tool allowlist.

The standalone config has no embedding store enabled by default, so `memory_search` uses the available BM25 path;
vector/BM25 fusion requires enabling the commented embedding components.

Invalid dates, missing credentials or services, invalid structured model output, unknown configured ETFs, missing
market files, and failed memory search stop the job. A market holiday is a successful skip. The workflow has no global
same-date execution lock or cross-file transaction, and repeated successful runs can resend DingTalk notifications.

## Tests

Focused unit tests mock model and market-data boundaries:

```bash
python -m pip install -e ".[dev,core]"
pytest tests/unit/test_auto_fin.py -v
```

Tests requiring real Tushare, LLM, or DingTalk credentials should be run separately and only with explicit
authorization.
