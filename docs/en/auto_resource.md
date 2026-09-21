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
