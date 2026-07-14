# Auto Dream

`auto_dream` is ReMe's long-term memory distillation flow from daily to digest. It scans daily inputs for a specified date,
processes only files that changed since the previous dream, extracts content worth retaining as memory units, integrates those
units into `digest/`, and generates the day's `interests.yaml` for proactive use.

<p align="center">
  <img src="../figure/auto-dream-and-proactive.svg" alt="ReMe Auto Dream and Proactive flow from daily to digest to proactive" width="92%">
</p>

Its daily inputs usually come from [Auto Memory](./auto_memory.md) and [Auto Resource](./auto_resource.md). For the file
semantics of `digest/`, `derived_from::`, and wikilinks, see [Memory as File](./memory_as_file.md). For the linking strategy
used during Integrate, see [Auto Link](./auto_link.md). To read `interests.yaml`, use [Proactive](./proactive.md).

## Configuration

The default configuration is in `reme/config/default.yaml`:

```yaml
auto_dream:
  backend: base
  parameters:
    date:
      type: string
      default: ""
    hint:
      type: string
      default: ""
    topic_count:
      type: integer
      default: 3
    topic_diversity_days:
      type: integer
      default: 7
  steps:
    - backend: dream_extract_step
      file_catalog: dream
      topic_session_id: interests
    - backend: dream_integrate_step
    - backend: dream_topics_step
      topic_count: 3
      topic_diversity_days: 7
    - backend: dream_finish_step
      file_catalog: dream
```

Parameters:

| Parameter | Purpose |
|---|---|
| `date` | Date to process in `YYYY-MM-DD` format. When empty, use today in the application's timezone. |
| `hint` | Additional guidance from the caller for the Extract and Integrate stages. |
| `topic_count` | Maximum number of topics written to `interests.yaml`. Defaults to 3. |
| `topic_diversity_days` | Number of past days of `interests.yaml` files considered when avoiding duplicate topics. Defaults to 7. |

## Inputs and Outputs

Inputs are daily Markdown files for the specified date:

```text
daily/<date>.md
daily/<date>/**/*.md
```

`daily/<date>/interests.yaml` is excluded from extraction input so topics from the previous run do not feed back into the
next extraction.

The main outputs are:

| Output | Description |
|---|---|
| `digest/procedure/*.md` | Methods, workflows, runbooks, and executable experience. |
| `digest/personal/*.md` | User-, team-, and project-related preferences, facts, and long-term context. |
| `digest/wiki/*.md` | General knowledge, concepts, observations, and decision precedents. |
| `daily/<date>/interests.yaml` | Topics worth proactive attention from the host agent that day. |
| `metadata/file_catalog/dream*` | Dream-specific catalog used to detect changes in daily inputs. |

## Four Stages

### 1. Extract

`dream_extract_step` performs three tasks:

1. Refresh the day's index page at `daily/<date>.md`.
2. Scan `daily/<date>.md` and `daily/<date>/**/*.md` and compare their mtimes with `file_catalog: dream`.
3. Send only changed files to the LLM and globally extract two structured result types: `units` and `topics`.

`units` are long-term memory units ready to be distilled into digest. Each has `name`, `bucket`, `summary`, and `paths`.
`bucket` may only be `procedure`, `personal`, or `wiki`; unknown values are routed to `wiki`.

`topics` are proactive-interest candidates for the day. They contain `title`, `reason`, `evidence`, `keywords`, and
`paths` and are filtered again in the Topics stage.

If there are no changed files, the flow ends early with success and skips later extraction work. If files changed but no LLM
is configured, Extract fails because extraction requires an LLM.

### 2. Integrate

`dream_integrate_step` invokes an agent independently for each unit and integrates that unit into one digest node. It exposes
these tools to the agent:

```text
node_search, read, frontmatter_read, write, edit, frontmatter_update
```

This stage carries the core responsibility of `auto_link`. It first uses `node_search` to recall similar or related nodes at
digest-node granularity, decides whether to create or update a node, and finally writes sources and related digest nodes as
wikilinks. See [Auto Link](./auto_link.md) for the recall, deduplication, and edge-writing rules.

There are four integration actions:

| Action | Meaning |
|---|---|
| `CREATE` | No equivalent abstraction exists; create a new digest node. |
| `CORROBORATE` | The same memory appeared again; append a source or strengthen the description. |
| `REFINE` | New material adds boundaries, steps, prerequisites, applicability, or detail. |
| `CORRECT` | New material corrects errors, omissions, or conflicts in the existing node. |

Successfully integrated units are recorded in `integrate_results`. Failed units enter `failed_units`, and their source paths
enter `failed_paths`. The Finish stage does not checkpoint failed paths, ensuring that they can be retried later.

### 3. Topics

`dream_topics_step` turns topic candidates from Extract into the final `daily/<date>/interests.yaml` for the day.

It reads:

```text
daily/<date>/interests.yaml
daily/<previous-date>/interests.yaml
```

Existing topics from the same day are preserved, while similar topics from the previous `topic_diversity_days` days are
deduplicated. At most three topics are written by default. With an LLM configured, the LLM selects topics that are more
specific, actionable, and non-repetitive. Without an LLM, the step falls back to local normalization and deduplication.

Example output format. See [Proactive](./proactive.md) for the interface that reads this file:

```yaml
date: 2026-06-20
topic_count: 3
diversity_days: 7
topics:
  - title: Quality regression in the memory retrieval pipeline
    reason: The user has recently made repeated changes to search, node_search, and dream integration.
    evidence: daily/2026-06-20/session.md
    keywords:
      - memory search
      - auto dream
    paths:
      - daily/2026-06-20/session.md
```

### 4. Finish

`dream_finish_step` completes the run:

1. Write successfully processed changed paths to `file_catalog: dream`.
2. Also write `daily/<date>/interests.yaml` and `daily/<date>.md` to the catalog.
3. Persist the dream catalog if there were upserts or deletions.
4. Return a summary containing counts for scanned, changed, integrated, topics, checkpoints, and related values.

Failed paths are not checkpointed. The next `auto_dream` run therefore continues to treat them as changed inputs until
integration succeeds.

## Running Auto Dream

CLI:

```bash
reme auto_dream date=2026-06-20
```

With caller guidance:

```bash
reme auto_dream date=2026-06-20 hint="Prioritize engineering decisions and long-term preferences"
```

The same set of steps can also be placed in a `cron` Job, for example to run every morning:

```yaml
jobs:
  daily_auto_dream:
    backend: cron
    cron: "30 3 * * *"
    steps:
      - backend: dream_extract_step
        file_catalog: dream
      - backend: dream_integrate_step
      - backend: dream_topics_step
      - backend: dream_finish_step
        file_catalog: dream
```

## Important Boundaries

`auto_dream` consumes only daily inputs and does not rewrite daily bodies. Daily preserves facts and the original situation;
digest is the abstracted long-term memory layer.

`digest` is not a copy of the source text. Its body should preserve reusable abstractions, while details point back to sources
through `derived_from:: [[daily/<date>/...]]`. Links follow the workspace-relative wikilink semantics described in
[Memory as File](./memory_as_file.md).

`auto_dream` does not invent an overview from nothing. Only content that actually appears in daily input and is extracted as
a unit or topic can enter digest or `interests.yaml`.

The complete flow depends on an LLM for Extract and Integrate. Topics can perform local deduplication without an LLM, but that
does not mean the full dream flow can run offline.
