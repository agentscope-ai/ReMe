---
name: Find a topic with memory tags
description: The tag field, normalization, file counts, and combined filtering.
memory_tags: ["aurora", "memory", "guide"]
---

# Memory tags

Tags live in the Markdown front matter's `memory_tags` list. ReMe derives its rebuildable tag index from files.

```yaml
memory_tags: [aurora, local_first, decision]
```

## Tags, directories, and links

Directories express file purpose. The `local_first` tag collects a topic across resources, daily, and digest. A [[digest/wiki/local-first.md|wikilink]] expresses a specific source or relationship.

## Try it

1. Select `local_first` in the search panel and inspect matches across directories and the file count.
2. Also select `release`: multiple tags match **any** selected tag, like the real search Job's tags filter.
3. Add “offline”: keywords and tags jointly constrain the results.
4. Edit the tags in [[digest/personal/communication.md|communication preferences]] and save; counts update immediately.

The demo accepts inline and indented lists, replaces whitespace with underscores, lowercases and deduplicates tags, and keeps up to three valid tags per file. It runs keyword filtering; ReMe's hybrid vector and BM25 retrieval requires a service.
