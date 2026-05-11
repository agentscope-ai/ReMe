# Memory Protocol

Single source of truth for vault schema, conventions, and the R-M-W
write loop. Consumed by:

- **Ingestor's embedded ReAct prompt** (`reme2/memory/ingestor.yaml` —
  injected as `{protocol}` at load time).
- **Strong-agent SKILL** (`reme-plugin/skills/reme/SKILL.md` —
  transcluded so the host agent sees the same rules).

Anything that defines schema invariants, path templates, write tool
semantics, or the R-M-W decision tree belongs here. Anything role-
specific (caller framing, audit trail expectations, summary
requirements) stays in the consumer.

## Vault layout

- **Topics** — long-lived cognitive memory at `topics/{folder}/{name}.md`.
  A **folder topic** has `folder == name`; it's the cluster's index head.
  Short wikilink `[[X]]` resolves to the folder topic if one exists,
  else falls back to a unique same-stem file.
- **Events** — fact log of one session at
  `events/{YYYY-MM-DD}/{name}/{name}.md`. The `.md` is the **index**
  inside a folder; sibling files are **materials** (raw conversation,
  tool outputs, data dumps). The index lists them under `## Materials`.

## Frontmatter — 4 schema axes

Every memory declares 4 orthogonal axes. The legacy `category` field is
auto-translated to these axes for back-compat reads, but new writes
should set the axes directly.

| Axis | Values | Meaning |
|---|---|---|
| `lifecycle` | `streaming` / `evolving` / `frozen` | streaming = events (decay/archive); evolving = topics (long-lived, edited); frozen = materials (immutable references) |
| `scope` | `instance` / `class` | instance = a specific moment / object; class = abstract concept / role / pattern |
| `source` | `auto` / `curated` / `derived` | auto = system-captured; curated = human/LLM intent; derived = computed from other memories |
| `role` | `observation` / `claim` / `question` / `profile` / `concept` / `method` / `reference` / `fundamentals` | cognitive role — drives ranking + role-specific validation |

### Conditional fields

- `confidence` ∈ {⏳, ✅, ❌} — **REQUIRED** when `role: claim` (legacy
  categories `thesis` / `model`). Same gate applies to `role: question`
  (legacy `questions`).
- `status` ∈ {`active`, `distilled`, `archived`} — meaningful only for
  `lifecycle: streaming`. Topic-style memories ignore it.
- `originSessionId` — should be set when `source: auto`.

### Standard identity fields

`title`, `description`, `tags`, `created`, `updated`, `topics`,
`parent`. Use today's date for `created` / `updated` on new writes.

## Cross-file references

`[[wikilink]]` syntax. Two forms:

- **Stem form** `[[X]]` — resolved against the file_store's stem index;
  prefers the folder topic if one exists.
- **Path form** `[[topics/X/X]]` or `[[topics/X/X.md]]` — anchored at
  the vault root.

## Link protocol — typed edges in body text

Edges live **only in body text**. YAML frontmatter is not walked for
links; a `links:` block in frontmatter is silently ignored by the
parser. Three legal inline forms:

| Form | Example | Predicate |
|---|---|---|
| Bare wikilink | `[[X]]` | `None` |
| Line-level Dataview | `extends:: [[X]]` | `extends` |
| Inline-bracketed Dataview | `[extends:: [[X]]]` | `extends` |

Multi-target on one field expands to multiple edges:

    concerns:: [[Topic A]], [[Topic B]]

Bullet markers are tolerated (`- extends:: [[X]]`, `* extends:: [[X]]`).

### Predicate syntax

A predicate is any **identifier-shaped token**: starts with a letter,
followed by letters / digits / underscore (regex
`[A-Za-z][A-Za-z0-9_]*`). The parser preserves whatever it sees — no
closed vocabulary at the schema layer. Choose predicates that read
well in prose (`extends`, `contradicts`, `derives_from`, `concerns`,
`supersedes`, …); the maintainer's lint pass can later surface
inconsistencies if a vocabulary policy is desired.

### `## Relations` section (machine-managed)

The maintainer's `discover` op appends newly discovered edges as
Dataview line-level fields under a `## Relations` heading at the end
of the file (creating the heading if absent). Example:

    ## Relations

    extends:: [[Source Topic]]
    concerns:: [[Topic A]]

`## Relations` is a **convention**, not a protocol — the parser treats
edges anywhere in the body uniformly. The heading is just the default
write location so machine-added edges stay separate from prose for
human review.

## Status state machine

`active → distilled → archived` (single direction, no skip). A reverse
or skip transition will be flagged by the Maintainer.

## Wikilink uniqueness

Every create path routes through `MemoryCreate.write`, which refuses to
introduce ambiguity (existing `[[X]]` would resolve to ≥2 paths). When
rejected, the response includes a `suggested_name`. Retry with that, or
pick a domain-specific qualifier (`Apple-Inc` beats `Apple-2`). Never
bypass with `force=true` unless you fully understand the ambiguity.

## Available tools

### Read tools (gather context BEFORE writing)

- `memory_get(path, include_chunks=False)` — full file content +
  frontmatter. On an event index, follow `## Materials` and read each
  artifact whose content you need.
- `memory_list(path_prefix=None, tags=None, metadata=None, limit=100)`
  — list indexed files filtered by prefix / tags / frontmatter.
- `memory_resolve_wikilink(wikilink)` — resolve `[[X]]` to a path;
  flags ambiguity / dangling.
- `memory_backlinks(path)` — files linking TO the given path.
- `memory_links(path)` — files the given path links to.
- `memory_search(query, …)` — hybrid (vector + keyword) chunk search.
- `memory_graph_search(query, seeds, graph_depth, …)` — vector +
  keyword + graph BFS fusion.

### Write tools (mutations are SSOT-routed; each returns success +
payload + records to audit)

- `memory_update(path, old_string, new_string, replace_all=False)` —
  body edit by exact-string substitution. Use a tail snippet to append.
- `memory_property_update(path, key, value)` — change one frontmatter
  key (`value=null` deletes). Use this to flip status.
- `memory_create(path, metadata, content, overwrite=False, force=False)`
  — new file. Reserve for genuinely NEW topics. Do NOT use for events
  (`sync` owns events). All paths must be ABSOLUTE under working_dir.
- `memory_rename(old_path, new_path)` — move file + rewrite cross-vault
  wikilinks. Refuses on destination conflict or stem ambiguity.
- `memory_delete(path)` — remove a file.
- `memory_archive(path)` — flip `status: archived` and move under
  `<vault>/Archive/`.

### Hot-write helper (deterministic, no LLM)

- `sync(name, description?, content?, topics?, tags?, materials?,
  on_date?)` — idempotent upsert of an event FOLDER per `(date, name)`.
  Reuse the same `name` across calls in one thread to keep extending
  the same folder. Refuses on `status: distilled` / `archived` and
  returns a `suggested_name`.

## R-M-W decision rules

Apply in order. Stop at the first match.

1. **SKIP** — if material is ALREADY covered by existing topics, reply
   with a single line `SKIP: <one-line reason>` and call no tools.
2. **CONTRADICT** — if material CONTRADICTS an existing block, use
   `memory_update` with a unique snippet of the outdated text and the
   corrected replacement.
3. **EXTEND** — if material EXTENDS an existing topic, use
   `memory_update` with a unique TAIL snippet of the existing body, and
   `new_string = tail + blank line + new content`.
4. **CREATE** — if material warrants a GENUINELY NEW topic, use
   `memory_create` at `topics/{folder}/{name}.md`. Do NOT
   `memory_create` under `events/` — `sync` owns that path.
5. **STATUS FLIP** — after integrating an event's content into a
   topic, flip that event's status to `distilled` with
   `memory_property_update`.

## Operating principles

- **Read before write.** Always inspect related topics before deciding
  CONTRADICT vs EXTEND vs CREATE. The wikilink-uniqueness gate refuses
  blind creates; reading first prevents wasted attempts.
- **Minimal edits.** Edit only what must change. Don't restructure
  while updating content.
- **Frontmatter on create.** Always include reasonable frontmatter:
  the 4 axes, `title`, `created`, `updated`, plus `confidence` when
  `role: claim` or `role: question`.
- **Never delete unless asked.** Distillation flips status; it does
  not remove events.
