# Auto Memory

Auto Memory is ReMe's entry point for conversational memory. Each conversation is first distilled into a daily memory card
identified by `session_id`, and the day's `YYYY-MM-DD.md` page then indexes all of those cards. It turns "we talked about it"
into "it was remembered" while preserving the original conversation as evidence.

<p align="center">
  <img src="../figure/auto-memory-resource.svg" alt="ReMe Auto Memory and Auto Resource writing daily memory cards" width="92%">
</p>

For the general file semantics of `daily/`, `session/`, frontmatter, and wikilinks, see
[Memory as File](./memory_as_file.md).

```text
Conversation
  ├─ step 1: daily/YYYY-MM-DD/<session_id>.md   # one card per conversation
  ├─ step 2: daily/YYYY-MM-DD.md                # daily index linking the cards
  └─ source: session/dialog/<session_id>.jsonl  # original conversation
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
      session-a.md
      session-b.md
```

`daily/2026-06-20/session-a.md` and `daily/2026-06-20/session-b.md` are memory cards distilled from different
conversations. `daily/2026-06-20.md` is the index page for that day. Resource files enter the same daily memory layer; see
[Auto Resource](./auto_resource.md).

When a call includes `session_id`, Auto Memory records that conversation separately under the given ID:

```text
daily/2026-06-20/session-a.md
```

This keeps different conversations separate. A requirements discussion, a debugging session, and a documentation update can
each have their own memory card. To see what happened on a particular day, start with `YYYY-MM-DD.md`. To inspect what was
distilled from one conversation, open the corresponding `<session_id>.md`.

## Preserving the Original Information

The distilled daily note is optimized for readability; the original conversation is retained for trust and verification.

While generating memory cards, Auto Memory also saves the raw sessions:

```text
session/
  dialog/
    session-a.jsonl
    session-b.jsonl
```

Each daily note points to its corresponding original conversation. When a memory needs verification, follow that link back to
the complete context in which it was created.

## Message Timestamps

Auto Memory preserves each message's `created_at` in both the prompt and the raw session JSONL. When importing historical
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

When a call does not explicitly provide `date`, Auto Memory uses the date of the earliest valid `created_at` value in the
messages. If no message contains a valid timestamp, it falls back to the current date. Historical imports may also specify the
target date directly:

```bash
reme auto_memory \
  session_id=locomo-session \
  date=2023-01-19 \
  messages='[{"role":"user","content":"Jon lost his job today."}]'
```

## What Happens Next

Auto Memory only creates memory in the daily layer. To distill this material further into long-term `digest/` nodes, use
[Auto Dream](./auto_dream.md). To search daily and digest content, use [Memory Search](./memory_search.md).
