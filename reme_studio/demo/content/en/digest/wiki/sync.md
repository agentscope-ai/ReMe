---
name: Synchronization scope
description: Distinguish deferred work from the accepted pilot scope.
memory_tags: ["aurora", "local_first", "planning"]
---

# Synchronization scope

Aurora's first pilot promises a single-device workspace, without automatic multi-device sync. [[daily/2026-09-15/decision.md|The local-first decision]] confirms this boundary.

## Future questions

Concurrent saves across devices, duplicate records, sensitive-data sync scope, and restoring previous versions still need designs. The current pilot did not validate these problems.

## Current operation

Users can download Markdown copies, but manual export is not a synchronization mechanism. Saving in this browser demo also belongs only to this browser and is not a backup of a real workspace.

See [[digest/wiki/release.md|release status]] for acceptance and [[digest/wiki/local-first.md|architecture knowledge]] for data-ownership principles.
