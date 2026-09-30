---
name: Changing to local-first
description: New evidence updates durable memory.
memory_tags: ["aurora", "local_first", "decision"]
---

# Decision diary · 2026-09-15

After [[resources/offline-feedback.md|field feedback]], Lin and Zhou approved local-first operation: Markdown files are durable sources, and indexes and graphs are rebuilt from files.

This updates yesterday's cloud assumption. We retain [[resources/requirements.md|the original draft]] and write the current conclusion, reasons, and sources into [[digest/wiki/local-first.md|architecture knowledge]].

Cross-device synchronization moves beyond the pilot; see [[digest/wiki/sync.md|sync scope]]. Model reasoning still depends on the configured provider. Local-first does not mean every model runs on the device.

Next: Zhou prepares offline reading and index recovery checks, while He observes field usability.
