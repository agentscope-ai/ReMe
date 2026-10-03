---
name: Local-first architecture decision
description: The current decision, reasons for changing it, and traceable sources.
memory_tags: ["aurora", "local_first", "decision"]
---

# Local-first

**Current decision: workspace Markdown is the durable source; indexes, catalogs, and graphs are rebuildable derived data.**

## Why the plan changed

[[resources/requirements.md|The September 14 draft]] proposed central cloud storage. [[resources/offline-feedback.md|September 15 field feedback]] revealed unreliable connectivity and restrictions on external sharing. [[daily/2026-09-15/decision.md|That day's decision]] approved local-first operation.

## What this means for users

Stopping the service does not prevent direct file reading. A damaged index should be rebuilt from source files; see [[digest/procedure/recovery.md|the recovery guide]]. If an online model is unavailable, model answers may be unavailable too. File ownership is unaffected.

Synchronization is planned separately in [[digest/wiki/sync.md|sync scope]]. This conclusion follows explicit evidence; do not mistake the early draft for the current plan.
