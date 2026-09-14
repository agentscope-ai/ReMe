# Auto Memory

Auto Memory is ReMe's entry point for conversational memory. Within a target date, it uses `session_id` to find or update at
most one daily memory card, whose filename is a concise topic or event name chosen by the Agent. The day's `YYYY-MM-DD.md`
page indexes those cards. It turns "we talked about it" into "it was remembered" while retaining a source conversation record
as evidence.

<p align="center">
  <img src="../figure/auto-memory-resource.svg" alt="ReMe Auto Memory and Auto Resource writing daily memory cards" width="92%">
</p>

For the general file semantics of `daily/`, `session/`, frontmatter, and wikilinks, see
[Memory as File](./memory_as_file.md).

```text
Conversation
  ├─ step 1: daily/YYYY-MM-DD/<generated_name>.md # one topic-named card per session
  ├─ step 2: daily/YYYY-MM-DD.md                  # daily index linking the cards
  └─ source: session/dialog/<session_id>.jsonl    # source conversation record
```

## What It Records

Auto Memory does not preserve a chat transcript as a running summary. It records information that may remain useful later:

- User preferences: preferred style, collaboration habits, and long-term requirements.
- Key facts: project background, important numbers, explicit conclusions, and constraints.
- Process decisions: what happened, why a choice was made, and which alternatives were rejected.
- Current state: what has been completed, what is blocked, and what comes next.
- Reusable experience: commands, workflows, diagnostic methods, and solutions.

## Write Location

Auto Memory writes distilled memories to `daily/`. Conversations from the same day first become individual cards:

Example directory:

```text
workspace/
  daily/
    2026-06-20.md
    2026-06-20/
      login-refactor-decision.md
      retrieval-regression.md
```

The two files under the date directory are topic-named cards distilled from different conversations.
`daily/2026-06-20.md` is the index page for that day. Resource files enter the same daily memory layer; see
[Auto Resource](./auto_resource.md).

When a call includes `session_id`, Auto Memory uses it to find the corresponding card through frontmatter, while the Agent
chooses a readable filename through `name`:

```yaml
name: login-refactor-decision
session_id: session-a
source_conversation: "[[session/dialog/session-a.jsonl]]"
```

This keeps different conversations separate without forcing opaque IDs into filenames. An update locates the existing note by
`session_id` or `source_conversation`; if the Agent supplies a better frontmatter `name`, the system can rename the note and
retarget inbound wikilinks. To see what happened on a day, start with `YYYY-MM-DD.md`.

## Preserving the Original Information

The distilled daily note is optimized for readability; a filtered source conversation record is retained for trust and
verification.

While generating memory cards, Auto Memory also saves the source messages:

```text
session/
  dialog/
    session-a.jsonl
    session-b.jsonl
```

Each daily note points to its corresponding conversation record. Saved messages omit tool-result blocks and base64 data
blocks, preventing recalled memory and binary payloads from being mistaken for user-provided evidence later.

## Optional Session Images

Images are disabled by default. Both boolean options, `include_images` and `supports_vision`, default to `false`.
To send images, enable the feature and explicitly declare that your selected model supports vision:

```bash
reme auto_memory session_id=session-a include_images=true supports_vision=true messages='[...]'
```

For a one-shot CLI invocation, use `reme start job=auto_memory` with the same arguments.
No YAML edit or rebuild is needed. `include_images` controls whether Auto Memory considers images;
`supports_vision` declares whether the AgentScope wrapper's bound `as_llm` supports images; it does not detect capabilities
or select a different model.
Only `include_images=true supports_vision=true` sends images. Choose a model and AgentScope formatter that support them.

`include_images=false` keeps the existing text-only behavior on every wrapper, regardless of `supports_vision`.
Text-only input still uses the existing string-based `agent_wrapper.reply()` path; other wrappers need no multimodal
interface changes. With `include_images=true`, the wrapper must be AgentScope; other backends raise
`NotImplementedError`, not a text fallback. `image_mode` defaults to `direct`, the only supported mode;
selecting any other mode while images are enabled raises `ValueError`.

When images are present but `supports_vision=false`, Auto Memory does not read them. It logs a warning and continues
with the original text-only input, reporting `status=fallback` and `reason=supports_vision=false` in
`auto_memory_images` metadata. An enabled call without image blocks reports `status=skipped`.

Pass AgentScope messages in `messages`, with top-level `DataBlock` images (`source.media_type` starting with `image/`).
Sources may be Base64, HTTP(S) URLs, or `file://` URIs inside the workspace. Local reads respect `_allowed_paths`.
Image inputs use bounded loading and provider preprocessing (including the existing 2048-pixel maximum side,
orientation correction and format conversion when needed). Original caller messages and files are unchanged.

### Direct multimodal input

With its standard history rendering, Auto Memory builds an AgentScope `UserMsg` with text and image `DataBlock` values
interleaved within the conversation. It keeps their original order and each message's speaker and timestamp boundaries,
including image-only messages and repeated block IDs. This describes the input ReMe passes to the wrapper; the selected
AgentScope formatter determines how that input is represented in provider requests.
The memory Agent extracts facts from text and images together, without separate caption calls. This is one Agent
workflow, not necessarily one API request: tool use can trigger further model turns that still include the images.
When no image-count limit is explicitly configured, direct mode reserves room for all incoming images in the Agent
context. An explicit `context_config.max_image_num` below the incoming count causes a visible whole-input text fallback,
not silent removal of older images. Model/provider context limits still apply.

Text-only and image-bearing Auto Memory calls use the same model bound through the AgentScope wrapper's `as_llm`.
`supports_vision=true` must describe that model. The optional `components.as_llm.vision` used for image resources
does not change Auto Memory's model.

Auto Memory does not create resource image files or separate caption cards. Relevant image facts may become part of
the normal daily memory note. The existing AgentScope wrapper also saves its internal Agent state under
`mem_session/agentscope`; in direct mode that state includes image inputs.

**Source JSONL saving is unchanged whether images are enabled or disabled.** No captions or image metadata are added to
it; the existing filtering above still applies. Replaying a saved JSONL cannot recover omitted
Base64 images. Resubmit the original image-bearing messages to process those images again.

If image loading or preprocessing fails, Auto Memory logs a warning and records the fallback in
`auto_memory_images` response metadata. It discards all temporary image enrichment for that request and continues with the
original text-only memory input, not a partially enriched conversation. The CLI can still succeed when that normal
memory operation succeeds; inspect the metadata to distinguish text fallback from successful image processing.
Cancellation, invalid enabled image-mode configuration, unsupported wrappers, and failures outside the image stage are
not swallowed by this fallback. Once the memory Agent starts, its failures propagate normally: ReMe does not rerun it as text, since its tools
may already have written memory. Direct mode reports `prepared` before the Agent runs and `completed` after it returns;
`captioned_images` is always zero for direct mode. `completed` describes image processing, not whether a note was written.

For persistent defaults, set `jobs.auto_memory.include_images` and `jobs.auto_memory.supports_vision` in the application
configuration (or the corresponding `reme start` overrides). For each option, call-time values override Job defaults,
which override Step configuration; both options default to `false`. No compilation is needed. This adapter targets AgentScope
messages, not Claude Code transcript parsing.

## Message Timestamps

Auto Memory preserves each retained message's `created_at` in both the prompt and the source conversation JSONL. When importing historical
conversations or benchmark data, provide the actual occurrence time for every message so the model does not confuse event
time with execution time:

```bash
reme auto_memory \
  session_id=locomo-session \
  messages='[
    {"role":"user","content":"Jon lost his job today.","created_at":"2023-01-19T08:00:00"},
    {"role":"assistant","content":"I am sorry to hear that.","created_at":"2023-01-19T08:01:00"}
  ]'
```

For compatibility with common dataset schemas, `auto_memory` also checks `time_created`, `timestamp`, `createdAt`,
`timeCreated`, and `created_time` when `created_at` is absent. These fields may appear either at the top level of a message
or inside `metadata`.

When a call does not explicitly provide `date`, Auto Memory uses the latest valid `created_at` date in the messages. If no
message contains a valid timestamp, it falls back to the current date. Historical imports may also specify the
target date directly:

```bash
reme auto_memory \
  session_id=locomo-session \
  date=2023-01-19 \
  messages='[{"role":"user","content":"Jon lost his job today."}]'
```

## What Happens Next

The default `auto_memory` and `auto_memory_cc` jobs run `auto_tag_step` after recording memory. Only a daily note that
was actually created or modified is tagged, using its final path after any rename. Claude Code callers still pass only
`session_id`; repeated Stop events with no new messages skip both memory generation and tagging.

Tags describe the document's central entities and are stored in the configured frontmatter key (`memory_tags` by default).
Per-file tagging failures are reported in `metadata.auto_tag` while preserving the memory response. Calls without note
changes do not automatically retry failed tagging; the existing file watcher updates the tag index asynchronously.

Auto Memory only creates memory in the daily layer. To distill this material further into long-term `digest/` nodes, use
[Auto Dream](./auto_dream.md). To search daily and digest content, use [Memory Search](./memory_search.md).
