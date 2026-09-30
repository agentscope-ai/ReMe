---
name: Offline and tag validation
description: Check file ownership, topic filtering, and recovery.
memory_tags: ["aurora", "local_first", "release"]
---

# Validation diary · 2026-09-17

Zhou stopped the local service and confirmed that Markdown files remained readable and copyable. After restarting, he checked derived-index recovery from existing files. Source files were not deleted to repair an index.

Lin filtered by `local_first` and found field resources, the decision diary, and architecture knowledge on the same topic. Adding “offline” narrowed irrelevant results.

He suggested explaining the distinction between tag filters and wikilink graphs. Tags collect a topic; links express specific relationships. Both are derived from file content.

We added [[digest/wiki/memory-tags.md|the tag guide]] and [[digest/procedure/recovery.md|recovery steps]]. Before release, check [[digest/procedure/release-checklist.md|the checklist]].
