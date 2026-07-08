# Overview

<p align="center"><em>Remember Me, Refine Me — a memory management toolkit for AI agents</em></p>

<p align="center">
  <img src="../figure/design-philosophy.svg" alt="ReMe Design Philosophy" width="92%">
</p>

ReMe turns conversations and resources into **readable, editable, and searchable
file-based long-term memory**. Long-term memory no longer hides inside a black-box
database — it lives as Markdown files in a workspace directory that both users and
agents can read, write, move, and delete.

```{note}
English documentation is in progress. The pages below mirror the Chinese structure;
some currently link back to the complete <a href="../zh/index.html">中文文档</a>.
```

## ✨ Core Ideas

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📄 Memory as File
Markdown files with frontmatter and wikilinks act as memory nodes that users and
agents can edit directly.
:::

:::{grid-item-card} 🌱 Self-evolving knowledge base
Auto Memory / Resource / Dream progressively distill conversations and resources into
long-term memory, weaving wikilink relationships automatically.
:::

:::{grid-item-card} 🔎 Progressive hybrid search
wikilinks + BM25 + embeddings combine keyword matching, semantic recall, and
relationship expansion.
:::

:::{grid-item-card} 🤝 Agent-friendly integration
`SKILL.md` + CLI integration lets different agents read, write, maintain, and reuse memory.
:::

::::

## 🔄 Memory Pipeline

ReMe's capabilities follow a **capture → consolidate → recall** pipeline:

- **Capture** — [Auto Memory](auto_memory.md) distills conversations into daily cards;
  [Auto Resource](auto_resource.md) interprets resource files.
- **Consolidate** — [Auto Dream](auto_dream.md) extracts and integrates daily notes into
  long-term `digest/`; [Auto Link](auto_link.md) weaves source and related wikilinks.
- **Recall** — [Memory Search](memory_search.md) does hybrid retrieval with link expansion;
  [Proactive](proactive.md) surfaces "what's worth attention today".

The underlying file model and runtime are covered in
[Memory as File](memory_as_file.md) and [Framework](framework.md).

## 📚 Start Reading

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Quick Start
:link: quick_start
:link-type: doc

Install, launch, and run your first write, index, and retrieval.
:::

:::{grid-item-card} 📄 Memory as File
:link: memory_as_file
:link-type: doc

The file-based memory model: layering, frontmatter, wikilinks, chunking.
:::

:::{grid-item-card} 🏗️ Framework
:link: framework
:link-type: doc

The Application / Service / Job / Step runtime and dependency injection.
:::

:::{grid-item-card} 🧠 Auto Memory
:link: auto_memory
:link-type: doc

How conversations become daily memory cards with preserved provenance.
:::

:::{grid-item-card} 🔎 Memory Search
:link: memory_search
:link-type: doc

Index building, hybrid recall, and progressive link expansion.
:::

:::{grid-item-card} ✨ Proactive
:link: proactive
:link-type: doc

Reading the day's interest topics to drive proactive reminders and insights.
:::

:::{grid-item-card} 📏 LME Benchmark
:link: lme_benchmark
:link-type: doc

Wire `context_answer_step` + `answer_judge_step` into a LongMemEval run: input/output
contract and an example config.
:::

::::
