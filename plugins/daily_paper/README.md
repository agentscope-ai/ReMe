# Daily Paper Plugin

[中文](README_ZH.md)

Daily Paper selects three papers from the Hugging Face Papers weekly and monthly rankings, downloads their arXiv PDFs,
and produces detailed Chinese reading notes plus a roughly five-minute Chinese brief. This directory is an independent
Python distribution. Its single `reme.plugins` entry point exposes a `plugin.yaml` containing five Step backends and
their Job configuration under `application_defaults`. Enable the installed plugin explicitly through
`plugins=["daily-paper"]`.

## Quick start

### 1. Install ReMe and Daily Paper

```bash
python -m pip install "reme-ai[core]>=0.4.1.9"
reme plugins install reme-daily-paper
```

### 2. Configure the model environment

Configure the LLM environment variables as described in the
[ReMe model-configuration guide](../../README.md#optional-model-configuration). Other compatible models and providers
can also be used. The workflow also requires network access to Hugging Face Papers and arXiv.

### 3. Start ReMe with the plugin

```bash
reme start plugins='["daily-paper"]'
```

With no explicit `config`, ReMe loads `default.yaml` and adds the plugin to that service. The plugin starts
`daily_paper_cron`, which runs daily at 08:00. From another terminal, generate a brief manually through ReMe's CLI
client:

```bash
reme daily_paper topics="Agent memory"
```

Or call its HTTP endpoint directly:

```bash
curl -s http://127.0.0.1:2333/daily_paper \
  -H 'Content-Type: application/json' \
  -d '{"topics":"Agent memory"}'
```

To run the Job once without starting a long-lived service:

```bash
reme start plugins='["daily-paper"]' job=daily_paper topics="Agent memory"
```

## Pipeline

```text
Hugging Face weekly/monthly rankings
                 ↓
merge ranks and exclude yesterday's and recently recommended papers
                 ↓
rank with RRF and let an Agent select three papers
                 ↓
download and parse arXiv PDFs, then write three Chinese analyses
                 ↓
use search + read to connect prior memory and generate a brief
                 ↓
refresh the daily index and optionally send the brief to DingTalk
```

`daily_paper_collect_step` concurrently reads the weekly and monthly rankings for the run date plus the strictly
preceding day's Daily Papers. It merges candidates by arXiv ID and excludes both yesterday's list and papers recommended
within `history_days`.

`daily_paper_rank_step` combines weekly and monthly positions with reciprocal-rank fusion and retains at most
`candidate_limit` papers. `daily_paper_select_step` then asks a tool-free Agent to select three unique candidate IDs.
Non-empty `topics` affect selection preference but not the fixed count.

`daily_paper_analyze_step` downloads PDFs into `resource/papers/`, reuses existing valid files, and extracts text within
the configured page, character, and file-size limits. It writes the three Chinese analyses in selection order. Scanned
PDFs and files without a text layer fail explicitly.

`daily_paper_digest_step` treats those three analyses as the factual source and receives only the read-only
`search` and `read` tools for linking earlier memory. Code validates historical wikilinks, appends links to all
three source notes, and rebuilds the daily index. The optional `dingtalk_markdown_send_step` sends the final brief when
conversation IDs are configured and otherwise skips without side effects.

## Parameters

| Parameter       | Default | Purpose                                                                           |
|-----------------|--------:|-----------------------------------------------------------------------------------|
| `date`          |    `""` | Empty uses today in the application timezone; otherwise use `YYYY-MM-DD`           |
| `force`         | `false` | Regenerate when that day's final brief already exists                             |
| `use_hf_mirror` | `false` | Use `HF_MIRROR_URL`, or `https://hf-mirror.com` when it is unset                   |
| `topics`        |    `""` | Optional topics to prioritize during selection                                    |
| `weekly_weight` |   `0.7` | Weekly contribution in reciprocal-rank fusion                                     |
| `history_days`  |    `30` | Prior recommendation window excluded by arXiv ID                                  |

Step-level defaults are `candidate_limit=20`, `rrf_k=60`, `hf_timeout=600`, `hf_max_retries=3`, `pdf_timeout=600`,
`max_pdf_bytes=52428800`, `max_pdf_pages=35`, and `max_pdf_chars=300000`.

The data clients automatically honor `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`. Manual runs enable the Hugging Face
mirror with `use_hf_mirror=true`; the cron Job enables it by default and can use the official service with
`DAILY_PAPER_USE_HF_MIRROR=false`. These environment variables override data sources and DingTalk settings:

```dotenv
HF_MIRROR_URL=https://hf-mirror.com
ARXIV_MIRROR_URL=https://export.arxiv.org
DINGTALK_APP_KEY=your-app-key
DINGTALK_APP_SECRET=your-app-secret
DINGTALK_ROBOT_CODE=your-robot-code
DINGTALK_CONVERSATION_IDS=cid-group-one,cid-group-two
```

## Output

```text
.reme/
├── daily/
│   ├── YYYY-MM-DD.md
│   └── YYYY-MM-DD/
│       ├── <Chinese-paper-title>.md  # three, kind: daily-paper-analysis
│       └── <Chinese-brief-title>.md  # one, kind: daily-paper-brief
└── resource/papers/
    └── <arxiv-id>.pdf
```

Markdown and PDF files are written atomically through temporary files in the same directory. `force=true` regenerates
the selected analyses and brief while reusing valid PDFs; it does not delete other notes already present for that day.
Network errors, too few candidates, invalid Agent output, and unparseable PDFs fail explicitly.

## Validation

```bash
python -m pytest plugins/daily_paper -v
```

Unit tests mock the Hugging Face, arXiv, AgentScope, and DingTalk boundaries and do not contact external services.
