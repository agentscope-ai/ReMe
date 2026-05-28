# Memory Protocol

Opinionated default contract for writing memory into a reme vault.
reme core reserves only `name` / `description` (both optional);
everything below is convention that consumers may replace.

## 1. Directory architecture

```
<vault>/
├── daily/
│   └── <YYYY-MM-DD>/
│       └── <slug>/
│           ├── <slug>.md            # hot summary note (markdown)
│           └── <material>.*         # sibling materials (any file type)
└── digest/
    └── <slug>/
        ├── <slug>.md                # cold canonical entry
        ├── <material>.md            # supporting docs
        └── <subslug>/               # nested narrower scope
            └── <subslug>.md
```

- **Hot tier** (`daily/…`) — streaming. One upstream writer per
  folder; every other consumer treats it as read-only. The
  summary note `<slug>.md` is markdown; siblings may be any
  file type the writer chooses.
- **Cold tier** (`digest/…`) — curated. Each folder is a
  scope and must contain `<folder>/<folder>.md` as its canonical
  entry. A scope's other children are sibling material files or
  narrower scope subfolders; nesting depth is unconstrained.
  Slugs are globally unique — a folder name appears at most once
  anywhere under `digest/`.

Facts flow one-way, `daily/` → `digest/`. References use the
full path relative to the vault: `[[digest/<slug>/<subslug>/<subslug>.md]]`.

## 2. Frontmatter

Reserved (typed; all optional):

| key | type |
|---|---|
| `name` | string |
| `description` | string |

Opinionated default axes (closed enums):

| key | values |
|---|---|
| `lifecycle` | `streaming` / `evolving` / `frozen` |
| `scope` | `instance` / `class` |
| `source` | `auto` / `curated` / `derived` |
| `role` | `profile` / `concept` / `claim` / `method` / `reference` / `observation` / `question` / `fundamentals` |

Any other keys consumers want (e.g. a workflow `status` flag) live
as extras — write them, read them with the `where` filter on `list`
tools (`null` matches absent-or-null); the protocol does not name
or enumerate them.

## 3. Body

Section structure is **advisory** — `## Summary`, `## Key Facts`,
`## Decisions`, `## Related` are convenient defaults but the
protocol mandates no specific section.

## 4. Wikilinks

Three recognized forms:

| Form | Example | Meaning |
|---|---|---|
| Bare | `See [[张三.md]]` | weakest layer — "mention" |
| Line-level Dataview | `colleague:: [[李四.md]]` | typed relation, queryable by predicate |
| Inline-bracketed Dataview | `主导 [负责:: [[项目X.md]]] 的重构` | typed relation, embedded inline |

Targets are stored **verbatim** as full paths relative to the vault.
`[[digest/zhang-san/zhang-san.md]]` resolves; short or
extension-less forms do not — no implicit `.md` completion, no
basename search, no folder-note expansion.

Renaming a node requires atomically rewriting every inbound
wikilink.

### 4.1 Typed predicates (half-open)

Recommended core vocabulary — writers may extend beyond this set,
but new predicates should be reused consistently:

| predicate    | meaning |
|---|---|
| `is_a`       | hierarchical (X is a kind of Y) |
| `part_of`    | containment (X is part of Y) |
| `depends_on` | dependency (X requires Y) |
| `manages`    | authority / responsibility |
| `alias_of`   | equivalence (X and Y are the same thing) |
| `references` | citation / external pointer |

Use typed `predicate:: [[X]]` only when the relation has clear
semantic weight; default to bare `[[X]]` for plain mentions. Typed
edges become queryable via `graph_traverse predicate=<name>`.

### 4.2 One-way write rule

A wikilink lives in the **source** node's body only — the node whose
prose introduces the relation. The **target** is never modified to
record the inbound relation. Backlinks are discovered at query time
via `graph_traverse direction=in`, never written into target bodies.

This keeps every write authoritative: a node's body reflects only
what its own author/writer chose to say, never sideways annotations
from other nodes' writers.
