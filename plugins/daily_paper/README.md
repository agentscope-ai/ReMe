# Daily Paper Plugin

[中文](https://github.com/agentscope-ai/ReMe/blob/main/plugins/daily_paper/README_ZH.md)

Daily Paper selects three papers from the Hugging Face Papers weekly and monthly rankings, downloads their arXiv PDFs,
and produces detailed Chinese reading notes plus a roughly five-minute Chinese brief. This directory is an independent
Python distribution. Its `daily-paper` entry point exposes five Step backends and complete Job defaults through the
package's `plugin.yaml`; enable it explicitly with `plugins=["daily-paper"]` after installation.

## Quick start

The workflow requires Python 3.11 or later, an available AgentScope LLM, and network access to Hugging Face Papers and
arXiv.

```bash
python -m pip install "reme-ai[core]>=0.4.1.8"
reme plugins install reme-daily-paper
export LLM_API_KEY="your-api-key"
export LLM_MODEL_NAME="qwen3.7-plus"
export LLM_BASE_URL="https://your-provider.example/v1"
reme start plugins='["daily-paper"]' job=daily_paper
```

The built-in LLM component defaults to:

- model: `qwen3.7-plus`
- endpoint: no built-in `LLM_BASE_URL`; set the OpenAI-compatible endpoint required by your provider
- environment variables: `LLM_API_KEY`, `LLM_MODEL_NAME`, and `LLM_BASE_URL`

Daily Paper uses the application's `default` LLM and `default` AgentScope wrapper. Select and Analyze call the wrapper
without tools, while Digest receives only the read-only `memory_search` and `read` ReMe Job tools.

The default workspace is `.reme/` beneath the process working directory. Override it with `workspace_dir=...` or the
application configuration you use alongside the plugin.

## Migrating from the built-in cookbook

Daily Paper is no longer imported or configured by the core `reme-ai` distribution. Install this package and enable
`daily-paper` explicitly instead of running `config=daily_cookbook` to obtain the Daily Paper Jobs:

```bash
# Before
reme start config=daily_cookbook job=daily_paper

# Now
reme plugins install reme-daily-paper
reme start plugins='["daily-paper"]' job=daily_paper
```

The workflow schemas have moved from `reme.schema.daily_paper` to `reme_daily_paper.schema`. The arXiv and Hugging Face
clients formerly under `reme.utils` are plugin implementation details under `reme_daily_paper`; applications that used
those modules directly must install this package and update their imports. Existing workspace notes, downloaded PDFs,
and indexes are not migrated or rewritten. Point the application at the same `workspace_dir` to keep using them.

## Pipeline

```text
Hugging Face weekly/monthly rankings
                 │
                 ▼
Collect ──► Rank ──► Select 3 ──► Analyze PDFs ──► Digest ──► DingTalk (optional)
                                      │               │
                                      ├─ PDFs         ├─ daily brief
                                      └─ paper notes  └─ day index
```

### 1. Collect

`daily_paper_collect_step` concurrently fetches:

- the Hugging Face weekly ranking for the run date's ISO week;
- the monthly ranking for the run date's calendar month; and
- Hugging Face Daily Papers for exactly the previous calendar day.

The weekly and monthly results are merged by arXiv ID while preserving both ranks. The step then excludes papers found
in yesterday's list or in the `arxiv_id` frontmatter of `daily/<date>/*.md` within the previous `history_days`.

If a Markdown file with `kind: daily-paper-brief` already exists and `force=false`, generation is skipped; the saved
brief can still proceed to DingTalk delivery. The job fails when no eligible papers remain.

### 2. Rank

`daily_paper_rank_step` uses reciprocal-rank fusion:

```text
score = 1 / (rrf_k + monthly_rank)
      + weekly_weight / (rrf_k + weekly_rank)
```

A missing rank contributes zero. Papers are ordered by fused score, upvotes, and arXiv ID. The pool is capped at
`candidate_limit`, and Rank applies no topic preference.

### 3. Select

`daily_paper_select_step` sends candidate metadata to a tool-free AgentScope agent and requires exactly three items:

```json
{"papers": [{"arxiv_id": "2601.01234", "reasoning": "A specific, verifiable reason"}]}
```

All IDs must be unique and belong to the candidate pool, and every reason must be non-empty. A validation failure is
returned to the agent for one retry. Only a non-empty `topics` value injects a personalized subject preference into the
selection prompt; it does not change the fixed count of three papers.

### 4. Analyze

`daily_paper_analyze_step` processes the selected papers in order:

1. validates a modern `YYYY.NNNN` or `YYYY.NNNNN` arXiv ID;
2. downloads the PDF to `resource/papers/<arxiv-id>.pdf`;
3. reuses an existing target whose header is `%PDF-`;
4. extracts paginated text with `pypdf`, bounded by `max_pdf_pages` and `max_pdf_chars`;
5. sends metadata, selection reasoning, and PDF text to a tool-free agent; and
6. writes a Chinese note to `daily/<date>/<Chinese-title>.md`.

Downloads use a temporary file and atomically replace the target only after validating the PDF header. They are also
bounded by `max_pdf_bytes`. There is no OCR fallback, so scanned or textless PDFs fail. When extraction is truncated,
the note records `pdf_text_truncated: true` in its frontmatter.

### 5. Digest

`daily_paper_digest_step` uses the three in-memory analyses as the factual source for the Chinese brief. It also
searches and, when needed, reads earlier daily notes to identify related coverage; those notes may only support
contextual wikilinks, not add facts about the current papers. The agent returns `title`, `desc`, and `body`. The code
then:

- strips model-generated YAML frontmatter if present;
- normalizes the Chinese title for use as a filename;
- keeps model-generated wikilinks only when they point to existing `daily/` Markdown files dated before the run date;
- deterministically appends wikilinks to all three source notes;
- writes `daily/<date>/<Chinese-brief-title>.md`; and
- rebuilds the `daily/<date>.md` day index.

Final response metadata includes the date, week/month scopes, selected arXiv IDs, selection reasons, note/PDF/brief
paths, source counts, and exclusion counts.

### 6. DingTalk

The final `dingtalk_markdown_send_step` is optional. With no conversation IDs it is a no-op. When configured, it strips
frontmatter and sends the brief body to each group in order:

```dotenv
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

A failed recipient does not prevent later attempts; the step reports a combined failure after trying every group.

## Outputs

```text
.reme/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <Chinese-paper-title>.md   # three, kind: daily-paper-analysis
│       └── <Chinese-brief-title>.md   # one, kind: daily-paper-brief
└── resource/
    └── papers/
        └── <arxiv-id>.pdf
```

Each successful generation writes three analysis notes and one brief. A forced rerun can leave unrelated or previously
selected analysis notes in the same day directory; ReMe does not delete them as cleanup. Filenames come from the agent's
Chinese titles. The implementation removes unsafe path characters and resolves title collisions. Markdown and PDF
outputs are written through same-directory temporary files and atomic replacement.

## Parameters and defaults

Public job parameters:

| Parameter       | Default | Purpose                                                                                 |
|-----------------|--------:|-----------------------------------------------------------------------------------------|
| `date`          |    `""` | Run date; empty uses today in the app timezone, otherwise requires `YYYY-MM-DD`         |
| `force`         | `false` | Regenerate even when the day's brief exists                                             |
| `use_hf_mirror` | `false` | Use the Hugging Face mirror from `HF_MIRROR_URL`, or `https://hf-mirror.com` when unset |
| `topics`        |    `""` | Topics to prioritize during selection                                                   |
| `weekly_weight` |   `0.7` | Weekly contribution to RRF                                                              |
| `history_days`  |    `30` | Prior recommendation exclusion window                                                   |

Step-level settings on the `daily_paper` job:

| Setting           |       Default | Purpose                                            |
|-------------------|--------------:|----------------------------------------------------|
| `candidate_limit` |          `20` | Maximum candidates sent to Select                  |
| `rrf_k`           |          `60` | RRF constant                                       |
| `hf_timeout`      | `600` seconds | Timeout for one Hugging Face request               |
| `hf_max_retries`  |           `3` | Maximum Hugging Face attempts                      |
| `pdf_timeout`     | `600` seconds | arXiv PDF download timeout                         |
| `max_pdf_bytes`   |    `52428800` | PDF limit, 50 MiB                                  |
| `max_pdf_pages`   |          `35` | Maximum extracted pages                            |
| `max_pdf_chars`   |      `300000` | Maximum extracted PDF characters sent to the agent |

## Mirrors

The data clients use httpx's default environment handling, so `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` take effect
when present. The two data sources reach a mirror differently: Hugging Face is gated on the `use_hf_mirror` job
parameter, while arXiv is driven by its environment variable alone.

```dotenv
# The plugin's daily_paper_cron Job enables the mirror by default; set false to use the official service
DAILY_PAPER_USE_HF_MIRROR=false

# Read only when the manual or scheduled job enables the mirror; defaults to https://hf-mirror.com when unset
HF_MIRROR_URL=https://hf-mirror.com

# Optional override; the code defaults to https://arxiv.org when unset
ARXIV_MIRROR_URL=https://export.arxiv.org

# Path-prefixed relay URLs are also supported
# HF_MIRROR_URL=http://relay-host:18080/hf
# ARXIV_MIRROR_URL=http://relay-host:18080/arxiv
```

`HF_MIRROR_URL` must implement the `/papers/...`, `/api/daily_papers`, and `/api/papers/...` routes used by the current
client. `ARXIV_MIRROR_URL` must implement `/pdf/<arxiv-id>`. A path prefix in either base URL is preserved, and a
trailing slash is optional. There is no fallback chain: whichever base URL a client selects is the only one it tries.

> **Behavior change:** `HF_MIRROR_URL` used to redirect Hugging Face traffic on its own. It is now read only when the
> job runs with `use_hf_mirror=true`; otherwise the official service is used and the client logs a warning that the
> variable was ignored. Pass `use_hf_mirror=true` for manual requests. The plugin's `daily_paper_cron` Job enables the
> mirror by default; set `DAILY_PAPER_USE_HF_MIRROR=false` to make that scheduled job use the official service.

## Running the workflow

Generate a brief for a specific date:

```bash
reme start \
  plugins='["daily-paper"]' \
  job=daily_paper \
  date=2026-08-06 \
  topics="Agent memory" \
  history_days=30
```

Force a rerun; valid local PDFs are still reused:

```bash
reme start plugins='["daily-paper"]' job=daily_paper date=2026-08-06 force=true
```

Start the HTTP service and scheduled jobs:

```bash
reme start plugins='["daily-paper"]'
```

With the default ReMe configuration, the HTTP service listens on `127.0.0.1:2333`. `daily_paper_cron` runs every day at
08:00 in the application timezone, prioritizes the topic `大模型长期记忆`, and uses the Hugging Face mirror by default.
Set `DAILY_PAPER_USE_HF_MIRROR=false` to use the official service. Override the service address through normal ReMe
configuration or startup arguments.

```bash
curl -s http://127.0.0.1:2333/daily_paper \
  -H 'Content-Type: application/json' \
  -d '{"date":"2026-08-06","force":false,"topics":"Agent memory"}'
```

## Failures and reruns

- Hugging Face HTTP failures use exponential backoff up to `hf_max_retries` attempts; invalid response payloads fail
  immediately.
- Fewer than three candidates, invalid agent selection, invalid/oversized/textless PDFs, or empty agent output stop the
  job.
- Papers are analyzed sequentially; PDFs and notes completed before a failure remain on disk.
- `force=true` regenerates the selected notes and the brief while reusing valid PDFs; it does not remove other notes
  already present in that day's directory.
- The multi-file workflow is not transactional and has no global per-date execution lock.

## Tests

The focused unit tests mock Hugging Face, arXiv, AgentScope, and DingTalk boundaries and do not call real services:

```bash
python -m pip install -e packages/reme_ai_studio -e ".[dev,core]" -e plugins/daily_paper
python -m pytest plugins/daily_paper -v
```
