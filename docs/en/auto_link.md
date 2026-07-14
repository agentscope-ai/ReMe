# Auto Link

In the current implementation, `auto_link` is not a separately registered Job. It is a capability of the Integrate stage in
`auto_dream`: when `dream_integrate_step` writes a memory unit to `digest/`, it also recalls digest nodes, makes a
deduplication decision, links sources, and weaves wikilinks to related nodes into the result.

For the complete dream flow, see [Auto Dream](./auto_dream.md). For general wikilink, frontmatter, and workspace-relative
path semantics, see [Memory as File](./memory_as_file.md). For question-answering retrieval, see
[Memory Search](./memory_search.md).

## Where It Runs

The default `auto_dream` flow is:

```yaml
auto_dream:
  steps:
    - dream_extract_step
    - dream_integrate_step   # where auto_link actually happens
    - dream_topics_step
    - dream_finish_step
```

The Integrate stage processes each unit independently. A unit is written to exactly one target digest node, but that node may
link to multiple sources and multiple related digest nodes.

## Goals

`auto_link` addresses graph quality at write time:

| Problem | Handling |
|---|---|
| The same memory already exists | Recall and update the existing node instead of creating a duplicate. |
| New and existing material are related | Write workspace-relative wikilinks into the body. |
| A digest node is disconnected from its sources | Point back to daily/resource source material with `derived_from:: [[...]]`. |
| A node contains only isolated prose | Add links to related digest nodes on both CREATE and UPDATE. |

## Toolchain

`dream_integrate_step` exposes these tools to the agent:

```text
node_search
read
frontmatter_read
write
edit
frontmatter_update
```

`node_search` is digest-only node retrieval designed for dream integration. It returns node-level signals such as the digest
node's `path` and the `name` and `description` from frontmatter. It does not expand the body and does not perform the link
expansion used by ordinary search.

`read` and `frontmatter_read` are used only for candidates that may be relevant, avoiding expansion of every recalled result
into a large context.

## Linking Flow

### 1. Recall candidate nodes

The agent first calls `node_search` with the unit's triggers, verbs, nouns, synonyms, and possible failure modes. Broad recall,
for example `limit=20-30`, is recommended by default because this step serves both deduplication and link discovery.

Recalled results are internally classified into three groups:

| Classification | Meaning | Next action |
|---|---|---|
| `same_abstraction` | The trigger or underlying abstraction is the same, with substantial content overlap. | Use as the UPDATE target. |
| `related` | An adjacent process, prerequisite, failure mode, concept, preference, or upstream/downstream knowledge. | Write a body wikilink. |
| `unrelated` | Only superficially similar or unrelated. | Ignore. |

### 2. Choose a write action

Every unit must select one action:

| Action | Linking semantics |
|---|---|
| `CREATE` | Write a new `digest/<bucket>/<slug>.md` and add source and related-node links to its body. |
| `CORROBORATE` | The same abstraction appeared again; append a new `derived_from:: [[...]]` and strengthen the description when needed. |
| `REFINE` | New material extends the existing node; insert the additional content in the appropriate section and preserve existing links. |
| `CORRECT` | New material corrects the existing node; use source links to identify the basis for the correction. |

An UPDATE should be additive whenever possible: do not delete existing wikilinks or `derived_from` entries. This prevents
later graph indexing and retrieval from losing edges.

### 3. Write source edges

Source edges use Markdown wikilinks:

```markdown
derived_from:: [[daily/2026-06-20/session.md]]
derived_from:: [[resource/2026-06-20/paper.md]]
```

These edges represent the evidence behind a digest node. Plain-text descriptions do not count as source edges because only
wikilinks can be parsed reliably by the file graph. For the complete parsing rules, see
[Memory as File](./memory_as_file.md#wikilink).

### 4. Write relationships between digest nodes

Relationships between digest nodes also use complete workspace-relative paths:

```markdown
relates_to:: [[digest/wiki/hybrid-search.md]]
depends_on:: [[digest/procedure/rebuild-index.md]]
blocks_on:: [[digest/personal/team-review-preference.md]]
```

Predicates are open-ended. Common forms include `relates_to::`, `depends_on::`, and `blocks_on::`. The predicate sits outside
the brackets, while the target path goes inside `[[...]]` and should include the `.md` suffix.

## Bucket Differences

`auto_link` adjusts the shape of its output according to the unit bucket:

| Bucket | Writing focus |
|---|---|
| `procedure` | Write a runbook with triggers, steps, inputs, and failure modes. Link prerequisites, substeps, and related preferences. |
| `personal` | Write user-, team-, or project-specific facts and preferences. Link related projects, habits, and decision context. |
| `wiki` | Write general knowledge, principles, observations, and decision precedents. Link concepts, methods, and adjacent knowledge. |

Regardless of bucket, preserve source edges and weave recalled related digest nodes into the body whenever possible.

## Relationship to Search

`auto_link` uses `node_search`, not the question-answering `search`.

| Capability | Purpose |
|---|---|
| `search` | External question answering; returns chunks and can expand upstream/downstream link context. |
| `node_search` | Dream integration; recalls only digest node-level summaries for deduplication and related-link decisions. |

This boundary matters. The Integrate stage needs to decide whether the same abstraction already exists and which nodes should
be linked; it should not load large numbers of body chunks into context. [Memory Search](./memory_search.md) handles
question-oriented chunk retrieval, RRF fusion, and link expansion.

## Failure and Retry

If integration of a unit fails, `dream_integrate_step` records `failed_units` and `failed_paths`.
`dream_finish_step` does not checkpoint those source paths, so the next `auto_dream` run processes them again.

This makes auto_link writes retryable: a failure does not mark the input as complete or silently discard digest edges that
should have been created.
