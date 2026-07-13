# ReMe Application Scenarios

This document describes how ReMe is used in real agent workflows. Directory names, Job names, and capability boundaries are
based on the latest code under `reme/`.

The common ReMe pattern is:

```text
Conversations / external resources
      |
      +--> auto_memory / auto_resource
      |        write to daily/
      |
      +--> auto_dream
      |        distill daily/ into digest/{personal,procedure,wiki}/
      |        and write daily/<date>/interests.yaml
      |
      +--> search / node_search / read / traverse / proactive
               let agents retrieve, associate, read, and inspect interest topics
```

## Scenario 1: A Supply-Chain Knowledge Base for a Financial Analyst

**Persona**: Analyst Wang, a new-energy industry researcher. Every day, Wang processes research reports, industry news,
company interviews, and spoken post-market notes.

**Pain point**: Information is scattered across text reports, web clippings, group messages, interview notes, and
conversations. A few days later, when asking, "How did the cobalt-price issue come up in the last CATL interview?", it is
difficult to reconnect the original event, company, material route, and upstream mining companies.

### Day 1: Post-market discussion and reports enter Daily

Analyst Wang synchronizes three reports to `resource/2026-05-18/`, then tells the agent:

```text
Glencore released its third-quarter report today, with cobalt output down 18% year over year.
We need to closely track how mining-rights policy changes in the DRC affect CMOC's KFM mine.
Downstream ternary-cathode manufacturers continue to move toward high-nickel, low-cobalt chemistry.
```

ReMe produces two kinds of lightly processed files:

```text
resource/
└── 2026-05-18/
    ├── glencore-q3.md
    ├── cobalt-policy.md
    └── cathode-trend.md

session/
└── dialog/
    └── 2026-05-18-close.jsonl

daily/
├── 2026-05-18.md
└── 2026-05-18/
    ├── 2026-05-18-close.md
    ├── glencore-q3.md
    ├── cobalt-policy.md
    ├── cathode-trend.md
    └── interests.yaml          # generated after auto_dream
```

The corresponding flow is:

- `auto_memory` saves the original conversation to `session/dialog/<session_id>.jsonl`, then asks the agent to write
  important facts to `daily/<date>/<session_id>.md`.
- `resource_watch_loop` watches text-file changes under `resource/` and triggers `auto_resource_step` to write a
  same-named daily note.
- `daily_create` maintains `daily/<date>.md` as the index page for that day.

### Day 1 evening: Auto Dream writes to Digest

Run:

```bash
reme auto_dream date=2026-05-18
```

`auto_dream` is a four-step pipeline:

```text
dream_extract_step
  scan daily/2026-05-18.md and changed files under daily/2026-05-18/
  output units and topics
dream_integrate_step
  recall existing digest nodes with node_search for each unit
  decide CREATE / CORROBORATE / REFINE / CORRECT
dream_topics_step
  write daily/2026-05-18/interests.yaml
dream_finish_step
  checkpoint successfully processed daily inputs
```

Outputs in this scenario:

```text
digest/
└── wiki/
    ├── glencore.md
    ├── cobalt.md
    └── ternary-cathodes.md
```

Example `digest/wiki/cobalt.md`:

```markdown
---
name: Cobalt
description: A key raw material for lithium-battery cathodes, with production concentrated in the DRC
---

downstream_product:: [[digest/wiki/ternary-cathodes.md]]
producer:: [[digest/wiki/glencore.md]]
source_event:: [[daily/2026-05-18/2026-05-18-close.md]]

# Cobalt

## Supply
Glencore's third-quarter cobalt output fell 18% year over year. Continue monitoring how tighter supply affects prices.

## Policy risk
Changes to mining-rights policy in the DRC may affect KFM mine operations and should be tracked together with CMOC.
```

Note that wikilinks use literal path semantics. Prefer complete workspace-relative paths with the `.md` extension. ReMe
does not automatically resolve `[[cobalt]]` to a particular file.

### Day 2: Interview findings refine existing nodes

Analyst Wang attends a CATL investor interview:

```text
CATL is switching fully to high-nickel 9-series ternary cathodes this year, so cobalt usage will keep falling.
Capacity utilization is 85%, five percentage points higher than last quarter.
```

`auto_memory` writes:

```text
daily/2026-05-19/catl-interview.md
```

During `auto_dream date=2026-05-19`:

- `dream_extract_step` extracts "CATL's switch to high-nickel ternary cathodes" and "CATL capacity utilization."
- `dream_integrate_step` uses `node_search` to recall `digest/wiki/ternary-cathodes.md` and
  `digest/wiki/cobalt.md` from `digest/`.
- The agent applies `REFINE` to `ternary-cathodes.md`, adding CATL's 9-series transition as a case.
- The agent applies `CREATE` to, or updates, `digest/wiki/catl.md`.

The graph gradually grows into:

```text
digest/wiki/
├── glencore.md
├── cobalt.md
├── ternary-cathodes.md    # REFINE: high-nickel, low-cobalt trend + CATL case
└── catl.md                # CREATE: capacity utilization + 9-series transition
```

### Day 5: The user searches for "upstream and downstream battery companies"

Analyst Wang asks:

```text
Help me analyze the upstream and downstream lithium-battery supply chain.
```

The agent calls:

```bash
reme search query="lithium battery upstream downstream ternary cathode cobalt CATL" limit=5
```

`search` returns chunk content, line numbers, scores, and outlink/inlink directories for matched files. With the default
configuration, results come from BM25 plus graph expansion.

The result shape is:

```text
========== digest/wiki/cobalt.md:8-20 [score=0.0148 keyword=3.7112] ==========
# Cobalt
## Supply
Glencore's third-quarter cobalt output fell 18% year over year...

  outlinks:
    -> digest/wiki/ternary-cathodes.md name="Ternary Cathodes" via predicate=downstream_product
    -> digest/wiki/glencore.md name="Glencore" via predicate=producer
  inlinks:
    <- digest/wiki/ternary-cathodes.md name="Ternary Cathodes" via predicate=upstream_material

========== digest/wiki/ternary-cathodes.md:5-18 [score=0.0139 keyword=3.2017] ==========
...
```

The agent can assemble a supply-chain outline from the neighbor directory alone. When it needs details, it can call:

```bash
reme read path=digest/wiki/catl.md
reme traverse path=digest/wiki/cobalt.md depth=2 direction=both
```

The final response might be:

```text
The lithium-battery chain can be divided into three segments:
1. Upstream raw materials: cobalt supply is concentrated in the DRC. Glencore is a major producer, and the policy impact
   on CMOC's KFM mine should be monitored.
2. Midstream materials: ternary cathodes continue to move toward high-nickel, low-cobalt chemistry.
3. Downstream batteries: CATL's move to 9-series high-nickel ternary cathodes confirms the downstream demand direction.

These conclusions come from the post-market conversation on 2026-05-18, the Glencore quarterly-report resource note, and
the CATL interview record on 2026-05-19.
```

### Proactive: Read the day's interest topics

`auto_dream` writes:

```text
daily/2026-05-18/interests.yaml
```

Example:

```yaml
date: 2026-05-18
topic_count: 3
diversity_days: 7
topics:
  - title: Impact of DRC mining-rights policy on cobalt supply
    reason: The user repeatedly mentioned KFM and cobalt-price risk today
    keywords: [cobalt, DRC, CMOC, KFM]
    paths:
      - daily/2026-05-18/2026-05-18-close.md
```

Call:

```bash
reme proactive date=2026-05-18
```

The `proactive` Job returns the topics from `interests.yaml` and, optionally, the raw YAML content.

### Value of this scenario

- The analyst focuses on reading materials and expressing judgments. ReMe writes facts to daily and distills long-lived
  concepts into digest.
- `node_search` lets dream find existing digest nodes before writing, preventing a new file for the same concept every day.
- Graph expansion in `search` lets the agent inspect structure before reading full content, reducing wasted context.
- Every conclusion is stored in Markdown and can be audited with an ordinary editor.

## Scenario 2: Cross-session Procedural Memory for a Coding Agent

**Persona**: Developer Zhang, who works on project issues over time in Claude Code, AgentScope, or other agents.

**Pain point**: The same kind of bug appears repeatedly, but the agent starts its investigation from scratch each time. The
user's coding style, testing habits, and project preferences exist only in the current conversation.

### First session: The build stalls

The user says:

```text
pnpm build stalls at 92%. CPU usage is low, but memory keeps growing.
```

The agent's investigation:

```text
1. Clear caches: no effect.
2. Upgrade the terser plugin: no effect.
3. Discover that fork-ts-checker is running out of memory.
4. Set NODE_OPTIONS=--max-old-space-size=8192: the build succeeds.
```

`auto_memory` writes:

```text
session/dialog/build-oom-2026-03-10.jsonl
daily/2026-03-10/build-oom-2026-03-10.md
```

After `auto_dream`, ReMe generates:

```text
digest/
├── procedure/
│   └── typescript-build-oom.md
└── personal/
    └── code-style.md
```

Example `digest/procedure/typescript-build-oom.md`:

```markdown
---
name: TypeScript project build OOM diagnostic path
description: When a build stalls and memory grows, check the type-checking process first
---

source_event:: [[daily/2026-03-10/build-oom-2026-03-10.md]]
related_preference:: [[digest/personal/code-style.md]]

# TypeScript Project Build OOM Diagnostic Path

## Symptoms
The build stalls near the end. CPU usage is low, but memory keeps growing.

## Preferred path
1. Check whether fork-ts-checker or another type-checking subprocess is running out of memory.
2. Try `NODE_OPTIONS=--max-old-space-size=8192` first.
3. Clear caches or upgrade the minification plugin only when there is specific evidence to do so.

## Known ineffective paths
- Deleting `.cache` alone did not resolve the issue on 2026-03-10.
- Upgrading the terser plugin did not resolve the issue on 2026-03-10.
```

Example `digest/personal/code-style.md`:

```markdown
---
name: User coding-style preferences
description: Engineering preferences repeatedly expressed by the user
---

# User Coding-style Preferences

## Comments
The user dislikes comments that restate what the code literally does. Comments should explain WHY or a complex constraint.

## Tests
The user prefers focused tests around the risk and dislikes broad, unrelated refactoring.
```

### Second session: Quickly recalling a similar problem

Six weeks later, the user asks:

```text
vite build also stalls during bundling. Is it the same kind of issue?
```

The agent first calls:

```bash
reme search query="vite build stalls memory growth TypeScript OOM" limit=5
```

Matches:

```text
digest/procedure/typescript-build-oom.md
daily/2026-03-10/build-oom-2026-03-10.md
```

The agent can skip low-value paths in its response:

```text
The previous similar issue was an out-of-memory failure in the TypeScript type-checking process. I suggest checking memory
during the build and the type-checking subprocess first, then trying NODE_OPTIONS=--max-old-space-size=8192. Clearing caches
and upgrading the minification plugin did not help last time.
```

### Value of this scenario

- `digest/procedure/` stores both "how to do it" and "which paths failed," letting the agent reuse diagnostic experience.
- `digest/personal/` stores user preferences so the agent can follow the same engineering style across sessions.
- The original conversation remains under `session/dialog/`; daily records stay traceable, and digest is only the
  long-term distilled result.

## Scenario 3: A Personal Second Brain

**Persona**: Engineer Li, who talks with an agent about work, books, family plans, running, and travel.

**Pain point**: Ordinary chat history accumulates chronologically. Three months later, it supports only full-text search and
struggles with associative questions such as "What was the book Alice recommended?" or "Why did I change my training plan?"

### Daily input

One day produces:

```text
daily/2026-04-20/
├── lunch-with-alice.md
├── running-plan.md
└── frontend-design-review.md
```

`auto_dream` extracts:

```text
digest/
├── personal/
│   ├── alice.md
│   └── exercise-preferences.md
├── procedure/
│   └── frontend-review-checklist.md
└── wiki/
    └── deep-work.md
```

Example:

```markdown
---
name: Alice
description: A friend of the user who often recommends reading material
---

recommended_book:: [[digest/wiki/deep-work.md]]
source_event:: [[daily/2026-04-20/lunch-with-alice.md]]

# Alice

## Reading recommendations
At lunch on 2026-04-20, Alice recommended a book about attention and deep work.
```

### An associative recall

The user asks:

```text
What was the book about attention that Alice recommended last time?
```

The agent can search first:

```bash
reme search query="Alice recommendation attention book deep work" limit=5
```

Matches:

```text
digest/personal/alice.md
  outlinks:
    -> digest/wiki/deep-work.md via predicate=recommended_book
daily/2026-04-20/lunch-with-alice.md
```

Then read:

```bash
reme read path=digest/wiki/deep-work.md
```

Final response:

```text
It was "Deep Work." The record shows that Alice recommended it at lunch on 2026-04-20, and you later categorized it under
attention and working methods.
```

### Value of this scenario

- daily preserves "what happened at the time."
- digest/personal records people, preferences, and long-term relationships.
- digest/wiki records books, concepts, and topics.
- Wikilinks connect "person -> book -> topic -> original event," which is closer to human recall than browsing chat history
  only by time.
