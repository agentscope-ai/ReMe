# ReMe: A Personal Knowledge Base That Keeps Growing After Every Conversation

Every day, we talk with AI.

It helps us analyze projects, read papers, and troubleshoot problems. We also tell it about our preferences, plans, and ideas we have not fully worked out yet.

But most of the time, when a conversation ends, its value is locked away in the chat history. The next time we open a new window, the AI may remember a conclusion but not where it came from. It may find an old conversation but fail to connect it with materials we read or decisions we made later.

Useful long-term memory should do more than preserve what once happened. It should keep organizing information, building connections, and bringing past knowledge back into future reasoning when needed.

That is exactly what ReMe sets out to do.

> **ReMe is a local-first, self-evolving personal knowledge base for AI agents. It continuously turns conversations and resources into readable, editable, searchable, and interconnected Markdown memories, while surfacing threads worth following.**

GitHub: [https://github.com/agentscope-ai/ReMe](https://github.com/agentscope-ai/ReMe)

Documentation: [https://reme.agentscope.io](https://reme.agentscope.io)

<p align="center">
  <img src="../figure/reme-blog/reme-blog-cover-benchmark.png" alt="ReMe self-evolving personal knowledge base and public benchmark results" width="100%">
</p>

## A Memory Loop That Keeps Growing

<p align="center">
  <img src="../figure/reme-blog/reme-blog-architecture.svg" alt="ReMe self-evolving memory loop" width="100%">
</p>

ReMe is not another chatbot, nor does it try to replace the agents you already use. It is a local memory layer that agents such as QwenPaw, OpenClaw, Hermes, and Claude Code can share.

Built around a set of ordinary files, it does four things:

- Auto Memory extracts information worth keeping from conversations;
- Auto Resource turns external materials into traceable memories;
- Auto Dream consolidates daily memories into long-term knowledge;
- Index, Search, and Proactive bring old memories back into new tasks.

Together, they form a `capture → index → consolidate → recall` loop:

- Conversations and external resources are preserved first;
- Valuable information is organized into daily memories;
- Scattered events are consolidated into long-term knowledge nodes;
- Search, knowledge links, and interest discovery bring old memories back into future reasoning.

Most importantly, this loop is centered not on an opaque database, but on files owned by the user. Indexes, graphs, and caches are merely derived state that can always be rebuilt.

## Memory as File: Your Memories Are Your Files

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-as-file.svg" alt="ReMe Memory as File" width="100%">
</p>

ReMe's core design is called **Memory as File, File as Memory.**

“Memory as File” means long-term memories are not hidden inside a product. They live in Markdown, JSONL, YAML, and original resource files within your workspace. You can open them directly in VS Code, Typora, or Obsidian, and back them up or move them with Git, cloud storage, or your own synchronization setup.

“File as Memory” means each file is more than plain text. With YAML frontmatter, section structure, line ranges, and Wikilinks, it becomes a memory node that can be indexed, connected, and continuously evolved.

For example, a long-term memory about writing preferences might look like this:

```markdown
---
name: "User preference: technical writing style"
description: Prefers stating the problem and outcome first, followed by technical details and examples.
kind: preference
---

The user wants technical articles to have a clear narrative and avoid unnecessary jargon.

When writing an article, refer to [[digest/procedure/Technical content writing process.md]].

## Sources

This preference was observed in [[daily/2026-08-07/content-discussion.md]], which records the user's writing guidance.
```

Months later, even if you have forgotten the conversation, the agent can still read the preference, find the related process, and follow `Sources` back to the original context.

This is also the key difference between ReMe and “black-box memory”: agents can organize memories, but users always retain the right to inspect, correct, move, and delete them.

## Auto Memory: Turning Conversations into a Daily Journal

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-memory.svg" alt="ReMe Auto Memory turns conversations into daily memories" width="100%">
</p>

A great deal of valuable information does not begin with “please remember this.”

For example, you might say in a conversation:

> “Let's not refactor the login module this week. We can do it after the customer demo. Upgrading dependencies directly caused compatibility issues last time, so let's add regression tests first.”

This short passage contains project status, a time constraint, a lesson from a previous failure, and a next action. Auto Memory extracts these details from the conversation stream and writes them into a daily memory card, while retaining a source conversation record in `session/dialog/`.

```text
session/dialog/project-a.jsonl                 Source conversation record
daily/2026-08-07/login-refactor-decision.md    Content-named memory card
daily/2026-08-07.md                            Daily index, providing an overview
```

`session_id` remains in the card's frontmatter for stable lookup and provenance; the filename comes from the Agent-generated
topic/event `name`, so it does not have to match the session ID.

The next time the login module comes up, the agent does not need to search through the entire chat history. It can immediately see why the refactor was postponed, what went wrong before, and what should happen next.

It is like having a recorder who is always present—not one that mechanically transcribes every word, but one that organizes what will still matter later.

## Auto Resource: Bringing External Materials into the Same Memory System

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-resource.svg" alt="ReMe Auto Resource turns external materials into traceable personal memories" width="100%">
</p>

Not all valuable information comes from conversations. Research materials, project documents, meeting notes, archived web pages, and structured data may all become part of a personal knowledge base.

Auto Resource provides a general entry point for external materials. After a resource enters `resource/`, ReMe preserves the original and organizes its topics, key facts, and actionable information into daily cards with `source_resource` links. It supports text resources including Markdown, plain text, JSON, JSONL, CSV, YAML, and HTML, plus image resources that a vision model turns into caption cards.

In other words, Auto Memory builds personal knowledge from conversations, while Auto Resource builds it from non-conversational materials. Both streams flow into the same daily memory layer, where ReMe indexes, consolidates, and retrieves them together.

### Daily Paper: An Example External-Resource Workflow

Daily Paper is an optional plugin built on this file-based memory system. It collects papers from the weekly and monthly Hugging Face Papers rankings, removes items recommended recently, ranks the remaining papers, selects three, saves their PDFs, and generates Chinese paper notes and a briefing that takes about five minutes to read.

Imagine that you regularly follow research on agent memory. Each morning, instead of receiving only three links, you get three detailed notes already saved locally. The briefing points to the original notes through Wikilinks, and each note links back to its PDF. A month later, when you ask, “What recent methods compress long-term memory?”, those materials are already in the same retrieval system. There is no need to search through browser history again.

Daily Paper demonstrates how Auto Resource can be composed into a concrete workflow, but the external-resource pipeline is not limited to papers.

## Auto Dream: Growing Daily Notes into Connected Long-Term Knowledge

<p align="center">
  <img src="../figure/reme-blog/reme-blog-auto-dream.svg" alt="ReMe Auto Dream extracts, classifies, and consolidates long-term knowledge from daily memories while adding Wikilinks" width="100%">
</p>

As daily notes accumulate, a new problem emerges: the information is all there, but it remains scattered across different dates.

Suppose conversations and external materials give you three pieces of information about the same problem:

- The first time a build hung, clearing the cache did not help;
- A project document later confirmed that insufficient Node.js memory was the root cause;
- A third note added that the issue occurs more often in large TypeScript projects.

By default, Auto Dream looks at the two most recent days ending at the target date and sends only daily files changed since
the previous run to extraction. It merges cross-file evidence for the same abstraction and keeps only the strongest reusable
memories within a default cap of five units, then writes them into three categories of long-term memory:

- `Personal`: preferences, conventions, and constraints specific to a user, team, or project;
- `Procedure`: repeatable processes, methods, and troubleshooting guides;
- `Wiki`: general definitions, principles, observations, and knowledge.

For example, the information above would become `digest/procedure/Troubleshooting frozen frontend builds.md`, which records the triggering conditions, diagnostic sequence, failed attempts, solution, and scope of applicability—instead of simply concatenating several daily notes.

When consolidating each memory unit, Auto Dream first searches existing nodes across `personal`, `procedure`, and `wiki`, distinguishing between the “same abstraction” and “related knowledge.” The same abstraction determines how the target node evolves:

- `CREATE`: no equivalent memory exists, so create a new node;
- `CORROBORATE`: the same conclusion appears again, so add its source and strengthen confidence;
- `REFINE`: new material adds conditions, steps, or details;
- `CORRECT`: new information corrects an earlier conclusion.

Related knowledge is written into the body as Wikilinks during the same consolidation process. This is Auto Link. For example, “Troubleshooting frozen frontend builds” can connect general knowledge, team preferences, and original evidence at once:

```markdown
This issue often occurs in [[digest/wiki/Large TypeScript projects.md]]. When resolving it,
follow the “add regression tests first” convention in [[digest/personal/Team change preferences.md]].

## Sources

The root cause and applicable scenarios were documented in
[[daily/2026-08-07/build-debug.md|Build troubleshooting record]].
```

Knowledge evolves and links are created in the same workflow. Relationships are not invisible edges hidden in a graph database; they are readable, editable content in the files themselves. The files can rebuild the graph—the graph never takes control of the files.

## Memory Index: Turning Ordinary Files into a Searchable Memory Network

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-index.svg" alt="ReMe Memory Index build process" width="100%">
</p>

Markdown is easy for people to read, but if files are merely piled into directories, agents still struggle to find them
quickly. The default live index watches Markdown under `daily/` and `digest/`. A separate resource workflow watches
`resource/` and turns those files into daily cards that enter the same index. Manual `reindex` rebuilds BM25 and
embedding indexes from the chunks those ingestion paths have already accepted; it does not rescan files or rebuild the
Wikilink graph.

A Markdown file is parsed into:

- One file node containing file-level information such as its path and frontmatter;
- Multiple semantic chunks split, wherever possible, along the boundaries of headings, paragraphs, lists, and code blocks, while retaining section structure and line numbers;
- Multiple Wikilink edges recording what the file points to and what points back to it.

For retrieval, ReMe can combine three types of signals:

| Retrieval signal | Problem it solves | Example |
|------------------|-------------------|---------|
| BM25 keywords | Exact names, terms, and identifiers must not be missed | “CATL”, “issue #184” |
| Embedding vectors | Semantically similar wording should still match | “build frozen” and “packaging stage not responding” |
| Wikilink graph | Reveal upstream and downstream relationships after finding a node | From “cobalt” to “ternary cathodes” and related research notes |

The default configuration enables BM25 and Wikilink expansion out of the box. Embeddings are optional and participate in vector retrieval only when enabled. Indexes, graphs, and caches are stored in `metadata/`; even if deleted, they can be rebuilt from the user's source files.

## Memory Search: Find the Answer First, Then Expand Relationships Progressively

<p align="center">
  <img src="../figure/reme-blog/reme-blog-memory-search.svg" alt="ReMe hybrid search and progressive expansion" width="100%">
</p>

Many RAG systems put all Top-K passages into the context at once. This is simple, but it creates two problems: isolated chunks lack context, while expanding every neighbor's full text quickly consumes tokens.

ReMe's hybrid search lets BM25 and optional vector retrieval produce their own candidates, then fuses the rankings with RRF. Instead of directly comparing BM25 scores with cosine similarities—two different scales—RRF combines where each result appears in the two ranked lists.

After retrieval, information expands progressively in three layers:

1. **Start with the matching passage**: return the most relevant chunk, file path, and line numbers;
2. **Then inspect the relationship directory**: show the file's outgoing and incoming links, including only each neighbor's path, name, description, and anchor rather than loading all of its content immediately;
3. **Finally, go deeper as needed**: the agent decides which relationship is genuinely relevant, then reads the original file or continues traversing the graph.

For example, you ask: “What was the name of the book about attention that Alice recommended last time?”

The first step may find a dinner note that says only, “The title contains the word ‘deep.’” The result also shows that the note links to Alice's personal node and is backlinked by reading notes for *Deep Work*.

The agent does not need to load Alice's entire profile, every reading note, and a whole month of journal entries into its context. It only needs to follow the most relevant link and read once more before answering:

> It was *Deep Work*. Alice recommended it at that dinner, and you later read Chapter 3 and left notes.

This resembles human association: first recall a fragment, then follow the trail to recover the full context.

## Proactive: Discovering Needs You Have Not Yet Put into Words

<p align="center">
  <img src="../figure/reme-blog/reme-blog-proactive.svg" alt="ReMe Proactive's two-way memory loop" width="100%">
</p>

At this point, ReMe has two input streams that continuously enrich the knowledge base:

- Auto Memory distills personal context from ongoing conversations;
- Auto Resource adds new knowledge from external materials.

Proactive reverses the direction. From accumulated conversations and materials, it discovers topics you have not yet resolved or may want to pursue, along with information you have not noticed but that closely relates to your recent work. These discoveries can then guide what external knowledge enters the system next.

For example, over the past week you separately mentioned that:

- Search results lack sources;
- Long documents lose section context after chunking;
- You want to compare several agent-memory evaluation methods.

Even though you never explicitly said, “Help me systematically study the explainability of memory retrieval,” Auto Dream can distill an interest topic from these daily memories:

```yaml
title: Evaluating the explainability of memory retrieval
reason: The user has recently focused on source tracing, structure-aware chunking, and memory evaluation.
evidence: daily/2026-08-07/search-discussion.md
keywords:
  - memory search
  - source attribution
  - benchmark
```

In a future beta release, after reading this topic through Proactive, a host agent could ask at an appropriate moment, “Would you like me to turn the retrieval issues we discussed recently into an evaluation plan?” It could also use the topic to initiate a user-authorized research workflow. Users would not need to identify and explicitly specify their interests and scope in advance; external resources related to needs implicit in their conversations could continue flowing into the knowledge base.

There is an important boundary: **ReMe's Proactive feature only reads and exposes interest topics. It does not independently access the internet, send notifications, or rewrite the knowledge base.**
It does not guess your interests from nowhere. It surfaces clues that already appeared in your behavior and conversations but have not yet been explicitly stated.

## Performance: Can It Retrieve Information from Very Long Histories?

ReMe uses LongMemEval and BEAM to evaluate memory across multiple sessions and extremely long conversations. During evaluation, the agent can use ReAct to search and read over multiple rounds, generate an answer, and then receive an LLM-as-judge score.

| Benchmark | Setting | Sample size | Agentic score | Primary capabilities tested |
|-----------|---------|------------:|---------------:|-----------------------------|
| **LongMemEval cleaned-s** | **Overall** | **500 questions** | **89.4%** | Cross-session retrieval, knowledge updates, and temporal reasoning |
| BEAM | 100K context | 20 cases / 400 questions | 66.1% | Ten types of long-context memory tasks |
| BEAM | 1M context | 35 cases / 700 questions | 65.0% | Larger-scale, ultra-long conversation settings |

LongMemEval cleaned-s includes single-session facts, preferences, multi-session reasoning, knowledge updates, temporal reasoning, and other question types. ReMe achieved an overall Agentic score of 89.4% across 500 questions. See the [LongMemEval evaluation guide](../../benchmark/longmemeval/README.md) for the complete workflow and breakdown.

BEAM covers ten categories of tasks, including contradiction resolution, event ordering, information extraction, knowledge updates, multi-session reasoning, preference following, summarization, and temporal reasoning. ReMe scored 66.1% on 20 cases / 400 questions with a 100K context and 65.0% on 35 cases / 700 questions with a 1M context. See the [BEAM evaluation guide](../../benchmark/beam/README.md) for the complete setup.

ReMe also uses $\pi$-Bench to evaluate the potential of multi-session reasoning to improve agent proactivity. The PROC score in $\pi$-Bench evaluates capabilities including directly fulfilling hidden intent, guiding targeted clarification, recovering cross-session preferences, reusing cross-session conventions, inferring cross-task dependencies, and advancing underspecified requests. Across five user personas, ReMe Agent achieved an average PROC score of 0.580, outperforming NanoBot by 2.4% under the same test-model configuration. See the [$\pi$-Bench paper](https://arxiv.org/abs/2605.14678) for details about the benchmark.

## Who Is ReMe For?

### People Who Use Agents Directly

If you want AI to understand you continuously throughout a long-term collaboration, ReMe lets your personal assistant stop starting from scratch. Your preferences, project context, important materials, and past decisions accumulate through ongoing conversations and can be found again when they are genuinely relevant.

Researchers, engineers, analysts, and other knowledge workers all fall into this category. Researchers can connect papers, discussions, and reading notes; engineers can preserve project decisions and cross-session troubleshooting experience; analysts can build an evolving record of events, perspectives, and sources. Their professions differ, but they share the same need: AI that can understand the past, accumulate experience, and recover supporting context for the next task.

### Developers Who Build Agents

If you are building an agent, harness, or AI product, ReMe provides an independent long-term memory layer. Through its CLI, HTTP API, MCP Server, or Python API, you can let multiple agents share the same file-based workspace without reimplementing memory extraction, knowledge organization, hybrid retrieval, and relationship expansion for every application.

Files remain the source of truth, while indexes and caches can be rebuilt at any time. This also makes it easier to determine whether an incorrect retrieval originated in the source material, memory consolidation, or the retrieval pipeline.

Ultimately, ReMe is for users and developers who want AI to do more than “answer this one request”: they want it to understand the past, accumulate experience, and know them better over the course of a long-term collaboration. We want agents to understand you better the more you use them—but that understanding should not live in a black box that you cannot inspect, correct, or take with you.

ReMe's answer is straightforward:

- Memories are files owned by the user;
- Original information preserves what happened, while long-term knowledge preserves the abstraction;
- New conversations and resources keep flowing in, while existing knowledge is continuously supplemented and corrected;
- Every conclusion can be traced to relationships and sources through Wikilinks;
- Indexes and caches serve the files rather than replace them;
- Agents can remember, organize, search, and discover, but users always retain ultimate control.

When these mechanisms come together, a personal knowledge base is no longer a repository you must maintain by hand.

It remembers a little more after every conversation and understands a little more after every new resource. At night, it reorganizes scattered experiences. When a future question arises, it follows the connections between pieces of knowledge and brings back the memory you actually need.

That is what ReMe sets out to do: **make memory not only persistent, but continuously evolving.**

## Integrate ReMe with the Agents You Already Use

ReMe can run as a local memory service accessed through its CLI, HTTP API, or MCP Server, or it can be embedded in a host
process through its Python API. The default HTTP service can also serve ReMe Studio at the same address for browsing,
editing, and searching the workspace and inspecting the digest wikilink graph. Different agents can choose the integration
that best fits their runtime environment and share the same local memory workspace when needed.

| Agent | Recommended integration | Capabilities after integration |
|-------|-------------------------|--------------------------------|
| **DeepSeek Harness** | Install [`@agentscope-ai/reme`](../../typescript/README.md#deepseek-harness) as a DSH profile bundle. | Long-term memory guidance, `reme_search`, automatic capture of completed main-agent turns, and scheduled Auto Dream. |
| **OpenClaw** | Install [`@agentscope-ai/reme`](../../typescript/README.md#openclaw) as the native memory plugin. | Recall before conversational root-agent runs, explicit search, automatic turn capture, and scheduled Auto Dream. |
| **QwenPaw** | Embed ReMe in-process through the Python API. | Reuse the host application's lifecycle and model configuration while keeping memories local and file-based. |
| **Claude Code** | Start the streamable HTTP MCP Service and install [`integrations/claude_code/reme`](../../integrations/claude_code/reme). | MCP memory-recall tools, the `reme-memory` skill, and a Stop hook that automatically records sessions. |
| **Hermes** | Start the HTTP Service and install [`integrations/hermes_agent`](../../integrations/hermes_agent). | Automatically recall relevant memories before model calls and invoke `auto_memory` asynchronously after each conversation turn. |
| **Codex and other CLI-capable agents** | Copy or install [`skills/reme_memory/SKILL.md`](../../skills/reme_memory/SKILL.md). | Search, read, and write memories through the CLI; automatic recording requires the host agent to integrate explicitly with the conversation lifecycle. |

For installation, configuration, and integration demos, see the [README](../../README.md).

## Contributions Welcome

ReMe is open source, and we welcome the community's help in making this self-evolving memory system more complete:

- Integrate more agents and harnesses so different runtime environments can use the same user-owned long-term memory;
- Contribute new Auto Resource sources and workflows so papers, news, and other public materials can continuously enter the knowledge base;
- Improve Auto Memory, Auto Dream, Auto Link, hybrid search, and Proactive so memories are organized more accurately, relationships are clearer, and retrieval is more reliable;
- Add application examples, evaluation tasks, and diagnostic reports to help us understand successes and failures in real long-term use;
- Improve documentation and tests, or share your needs and ideas for personal AI memory through an Issue.

Whether it is a code contribution, a use case, a bug report, or a new memory workflow, every contribution can bring ReMe closer to a truly readable, controllable, and continuously evolving personal knowledge base.

Contribution guide: [https://docs.agentscope.io/reme/latest/en/contribution](https://docs.agentscope.io/reme/latest/en/contribution)
