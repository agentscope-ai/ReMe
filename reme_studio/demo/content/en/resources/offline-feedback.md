---
name: Offline field feedback
description: Unreliable site connectivity changes the deployment assumptions.
memory_tags: ["aurora", "local_first", "resource"]
---

# Field feedback · 2026-09-15

He visited two pilot sites. Site A lost connectivity in a meeting room for about 20 minutes. Site B cannot send internal material to an external service. Both need to read existing records while offline.

## Observations and constraints

An online model may still perform reasoning tasks, but basic file reading, editing, and retrieval must not assume public internet access. Users need to inspect exactly which material they have saved.

## Recommendation

Use local workspace files as durable sources and treat indexes as rebuildable data. Make cross-device synchronization a later, explicit option so it does not block this pilot.

This recommendation was approved in [[daily/2026-09-15/decision.md|the decision diary]] and integrated into [[digest/wiki/local-first.md|durable architecture knowledge]]. It revises [[resources/requirements.md|the draft's cloud assumption]].
