# Auto Resource `Beta`

Auto Resource is ReMe's entry point for interpreting resources and is currently in **Beta**. Resource files first enter
`resource/`, preferably under a date directory, and are then interpreted into daily resource cards. Each card's filename
comes from the LLM-generated frontmatter `name`, and `source_resource` links the card back to its original file.

<p align="center">
  <img src="../figure/auto-memory-resource.svg" alt="ReMe Auto Memory and Auto Resource writing daily memory cards" width="92%">
</p>

For the general file semantics of workspace layers, `resource/`, and `daily/`, see
[Memory as File](./memory_as_file.md). For the flow that writes conversations to daily, see
[Auto Memory](./auto_memory.md).

```text
resource/[YYYY-MM-DD/]<resource_file>
  ├─ step 1: daily/YYYY-MM-DD/<generated_name>.md # interpreted resource card
  ├─ step 2: source_resource points to the original resource
  └─ step 3: daily/YYYY-MM-DD.md                  # daily index linking the cards
```

## What It Records

Auto Resource does more than copy file content. It extracts information that will make the resource easier to retrieve
and understand later:

- Core content: what the resource is mainly about.
- Structure: its sections, tables, fields, and data organization.
- Key details: important numbers, names, dates, and conclusions.
- Context and purpose: why the resource exists and how it relates to current work.
- Actionable items: tasks, deadlines, and follow-up work.

In short, it turns "a file was archived" into "the resource is usable."

## Original Resource Entry Point

Auto Resource uses `resource/` as the entry point for source material. Date directories are recommended, and their date
determines which daily memory layer receives the interpreted card. A file directly under `resource/` is also supported
and uses today in the application timezone when it is first processed. On later days, an exact `source_resource` match
keeps updates and deletion tied to that original daily card instead of creating a new card or leaving an orphan.

Example directory:

```text
workspace/
  resource/
    quick-note.txt             # enters today's daily layer
    2026-06-20/
      market-report.md
      meeting-notes.csv
```

Text resources such as `md`, `txt`, `json`, `jsonl`, `csv`, `yaml`, and `html` are the primary fit. Image resources
(`png`, `jpg`, `jpeg`, `webp`, `gif`, `bmp`, `tiff`, `heic`) produce caption cards as described in
[Image Resources](#image-resources).

Internally, one `AutoResourceStep` receives each change batch and sends every item to the first configured processor
whose class-level matcher accepts it. `AutoImageResourceStep` handles image suffixes and `AutoTextResourceStep` is the
final fallback. A new modality can therefore add a registered processor, its prompt, and one `dispatch_steps` entry
without changing the router.

## Resource Retention and TTL

The default configuration retains original resources. `config=proactive` enables a daily cleanup Job without a catalog,
watcher, or resource acknowledgement. Keep permanent files outside `resource/`.

```yaml
jobs:
  resource_cleanup_cron:
    backend: cron
    cron: "0 3 * * *"
    steps:
      - backend: cleanup_resources_step
        ttl_days: 7
```

At 03:00 in the application timezone, cleanup recursively removes regular files under `resource_dir` whose mtime is at
least seven days old, then removes empty directories while retaining the root. All file types and directory names are
eligible; filename dates do not determine expiry. Daily scanning can retain an expired file for up to another 24 hours.
Symlinks are skipped, and configured roots must stay inside the workspace without symlink aliases. Cleanup requires
symlink-safe descriptor-relative operations; unsupported platforms fail without deleting files. Failures are reported in
`answer.errors`, alongside `deleted_files`, `deleted_dirs`, and `freed_bytes`. Partial failures fail the Job and can be
retried. Disable physical cleanup with `jobs.resource_cleanup_cron.enabled=false`.

This replaces the proactive catalog, `resource_watch_loop`, `resource_changes`, `resource_processed`, and date-directory
TTL. Existing files need no conversion: their actual mtimes determine retention. The default Auto Resource indexing
pipeline continues to use its own catalog.

## Connector Tools and Screen Capture

Start the service in one terminal:

```bash
reme start config=proactive
```

Call tools from another terminal:

```bash
reme screen_capture
reme finance_news
reme check_json path=reme_metadata/pa/proactive.json
```

`BaseConnectorStep` returns `answer={"context": str, "paths": list[str]}`. `context` contains result context and reading
instructions; paths are workspace-relative files or directories for progressive inspection. Text-only results may return
an empty path list. No `metadata.json` is generated. Callers that retain task results should retain both fields together.

`screen_capture` compares displays by stable identity and saves changed original-size PNGs under:

```text
resource/screen_capture/2026-11-07_10-20-30-123456_display_<identity-hash>.png
```

The filename token hashes the native display identity; it is not an enumeration index. macOS uses display UUIDs;
other backends use MSS's optional `unique_id`. Missing or ambiguous identities always warrant analysis. PNG metadata
retains identity, geometry and comparison exclusions; original pixels are unchanged. Missing, unreadable, or differently
sized baselines also warrant analysis. Manual calls compare against the last successfully published manual capture.
When all screens are similar, the tool returns an explanation and `paths=[]`, without publishing files.

Comparison settings live in `components.screen_capture.default`: longest edge 1280, RGB channel difference greater
than 20, global changed-pixel ratio 0.0005, and overlapping 32×32 tiles at stride 16 with ratio 0.015. Either ratio reaching
its threshold warrants analysis. On macOS, floating windows titled `Remy · ReMe` and the known clock region are excluded
from comparison; both old and new window positions are considered. The original screenshots still include these regions.
Other desktop changes, including caret blinking and hover UI, may warrant analysis without being worth a reminder.

The lifecycle-owned component permits one capture at a time and rejects concurrent requests. Native APIs run in a child
process with a 90-second default deadline; timeout, cancellation, stop, and shutdown terminate and reap the child.
Comparison and publication run off the service event loop. Capture failures are errors, never unchanged results.

The textual context includes the capture time with timezone, display count, and interpretation instructions. Displays are
captured sequentially. Files are staged before publication;
a failed or cancelled collection removes its own published files without overwriting existing captures. Unlike the old
resource-package layout, publication of multiple flat files is not a single atomic directory operation.

Scheduled screenshot interpretation is a query in `{metadata_dir}/pa/cron.json`, independently dispatched by a ReMe
CronJob through `cron_agent` every five minutes. The agent calls the screenshot tool as needed and may omit a new
summary when nothing useful has changed. Execution and summary queries still run; no task configuration skips the whole
model round through preflight. Disable the task with `enabled: false`; file edits reload automatically.
Cumulative summaries live under `{metadata_dir}/pa/cron/screen_activity/YYYY-MM-DD/` for proactive consumption.
Install `reme-ai[core]` and grant screen-recording permission to the process
running ReMe. Availability depends on the local MSS backend.

`finance_news` fetches the last 24 hours of CLS telegraph news, deduplicates against `news_id` values in retained JSONL
files, and saves only newly observed items in at most one file per call:

```text
resource/finance_news/2026-11-07_15-30-00-123456.jsonl
```

Each line follows [`FinanceNewsItem`](../../reme/schema/finance_news.py): `news_id`, timezone-aware `event_time`, `source`,
`title`, and complete HTML-cleaned `content`. `context` gives the new-item count, query window, and reading instructions;
read the returned `paths` for the news itself, in batches as needed. The first call saves all valid news in the window.
Later calls, including after restart, save only IDs absent from retained files. Revisions with the same ID are not new
items. No new items means an explanatory context and `paths=[]`, without creating a file. The 24-hour window applies to
queries; historical files follow resource retention. Deleting history also removes its deduplication evidence.

Collection and cleanup share an in-process path lock to prevent duplicate concurrent publication. Pagination must cover
the full window. Defaults are a 10-second wait after each request, up to three attempts per page, a 5-second connection
timeout, and a 20-second request timeout. Step configuration accepts `request_interval` and `max_retries`; the tool takes
no arguments. Failure or cancellation publishes no partial results and leaves existing files intact. Corrupt, oversized,
or symlinked history fails explicitly. An error must not be interpreted as no new news.

`check_json(path)` parses a complete file without modifying it. Success returns the parsed value, including scalar
values and `null`. Failure returns `success=false` with `answer={path, line, column, error}`; non-parse failures use null
line/column values. Oversized files fail rather than being parsed after truncation. Configuration-specific validation,
such as cron expressions and active hours, remains the configuration loader's responsibility.

## Image Resources

Text and image resources share the same agent-wrapper and note-writing tools. Image inputs add a native AgentScope
image block alongside the interpretation instructions; the agent writes a caption card linked to the original image.
The card body starts with an `![[resource/...]]` embed link and the frontmatter carries `kind: image` and `media_type`,
so text search reaches image content through the caption.

Image processing is enabled by default (`include_images=true`) and requires an AgentScope wrapper bound to a compatible
model and formatter. Configure the model through `components.agent_wrapper.<name>.as_llm`, selecting the wrapper with
`agent_wrapper` on the resource Step. The former image-Step `as_llm` override and automatic `as_llm.vision` selection
are replaced by that binding. There is no separate caption model, schema-extraction call, or text-only retry after an
agent failure. An agent workflow can make multiple model requests while using its tools.

Each image interpretation starts a new session. Use the returned `agent_session_id` to find its processing record;
reprocessing the same image still updates the original card. The body should contain the image embed followed by a
description or transcription under `## Caption`, not an empty caption or a JSON response. Leave `status` to later
processing steps and keep its existing value when updating the card.

Customize image instructions with `prompt_dict.resource_instructions` (`resource_instructions_zh` for Chinese).
Rename existing `user_message` / `user_message_zh` settings accordingly.
Shared create/update templates insert these instructions at `{resource_instructions}`; older templates
without the placeholder receive them at the end.

Set `include_images=false` on an `auto_resource` call or as a Job default to skip **all** image events, including
deletions. Call-time values override Job defaults; when neither is set, image processing is enabled. For the watcher, use
`jobs.resource_watch_loop.include_images=false`; for manual calls, use `jobs.auto_resource.include_images=false`.
The image processor reports each skip in the existing result and warning log; text processing is unchanged. Existing
image cards are left untouched, even if their source image is deleted. Re-enabling images does not replay skipped
events; explicitly submit the affected paths to `auto_resource` when compensation is needed. The wrapper's configured
image-count limit is respected and must allow at least one image per resource call; it is not increased automatically.

Configure the wrapper when starting the persistent service. For example, to allow one image per agent context:

```bash
reme start components.agent_wrapper.default.context_config.max_image_num=1
```

The watcher processes resource changes automatically. To explicitly reprocess an existing `resource/photo.png`, run
the client in another terminal using the same workspace:

```bash
reme auto_resource include_images=true changes='[{"path":"resource/photo.png","change":"modified"}]'
```

Images wider or taller than 2048px are downscaled,
and provider-unfriendly formats are re-encoded, in memory for the request only; the original file under
`resource/` is never modified. Before a full decode, image dimensions are checked against a default limit of 40,000,000
pixels; images over the limit and Pillow decompression-bomb warnings fail only that resource. EXIF orientation is
applied to the in-memory request copy before resizing or conversion. Oversized JPEGs first use decoder-level
downsampling, followed by a final thumbnail pass when needed. The VLM request MIME and the card's frontmatter
`media_type` use the format Pillow detects from the image bytes, rather than trusting the filename extension. When an
image changes, its card is updated in place; when the image is deleted, the card is removed with it, provided image
processing is enabled.

Image preprocessing uses Pillow from the `core` extra. HEIC resources additionally require the optional
`image-heif` extra: `pip install "reme-ai[image-heif]"`. Other supported image formats do not load or require the HEIF
plugin.

## Resource Cards

Each resource file produces one daily resource card. The system initially uses the resource file's stem as a temporary
path. After the matching processor writes the card, the file is renamed according to its frontmatter `name`:

```text
resource/2026-06-20/market-report.md
        ↓
daily/2026-06-20/market-report-highlights.md
```

The resource card links to the original file through frontmatter:

```yaml
source_resource: "[[resource/2026-06-20/market-report.md]]"
```

When a resource changes, Auto Resource finds and updates the corresponding card through an exact `source_resource`
match. When an enabled resource is deleted, only the explicitly linked daily note is removed. A same-stem note without that
provenance marker is treated as user-owned and left untouched; new resource cards use a collision-free path instead.

A failed call may still have changed a card; `modified` records whether the file changed. If the agent writes the card
and then fails or is cancelled, the written content stays on disk. ReMe tries to complete metadata and update the day's
index for the card linked through `source_resource`, while preserving the original error or cancellation. A failed
image-note format check also leaves the written content in place. Failed calls are not retried automatically.

## Daily Index

Resource cards enter the same daily memory layer as Auto Memory cards. The day's `YYYY-MM-DD.md` page acts as an index
and organizes those resource cards:

```text
daily/
  2026-06-20.md
  2026-06-20/
    market-report-highlights.md
    meeting-notes-summary.md
```

To review which resources were processed on a day, start with `YYYY-MM-DD.md`. To inspect what was distilled from one
resource, open its corresponding resource card.

## Preserving the Original Resource

The interpreted daily note is optimized for readability; the original resource is retained for trust and verification.

Auto Resource does not move the original file. It remains at its original path under `resource/`. Resources can
therefore enter the daily memory flow while their source files stay in their original location.

## What Happens Next

Auto Resource only creates resource interpretations in the daily layer. To distill long-term knowledge from resources
into `digest/`, use [Auto Dream](./auto_dream.md). The default live index covers daily cards and digest nodes. Manual
`reindex` only rebuilds search indexes from chunks already accepted by an ingestion path; it does not add the original
resource files to search. See [Memory Search](./memory_search.md).
