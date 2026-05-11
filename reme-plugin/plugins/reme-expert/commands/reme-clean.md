---
description: Vault hygiene sweep — finds broken wikilinks, schema violations, stem collisions, stale events; proposes / applies fixes via the reme-curator subagent.
argument-hint: [target_prefix] [dry_run=true|false]
---

Spawn the `reme-curator` subagent for a vault hygiene sweep. The subagent runs in its own context window and returns a short audit; the main session stays focused.

Default behavior is `dry_run=true` — report the proposed plan first. Only re-run with `dry_run=false` after the user confirms the plan looks right.

Use the Agent tool with `subagent_type: reme-curator` and the following prompt:

```
Vault hygiene sweep.

Args: $ARGUMENTS

Defaults if not specified in args:
  - target_prefix: "" (whole vault)
  - dry_run: true
  - ops: ["lint", "decay"]

Steps:
1. memory_lint to enumerate broken wikilinks, schema violations, stem collisions.
2. memory_list filtered by status=active + old `created` for decay candidates.
3. Propose a fix plan grouped by op (rename / archive / property fix / dangling).
4. If dry_run=false, apply the plan via memory_rename / memory_archive / memory_property_update.
5. Return: scanned N files, found K issues across [categories], applied L fixes (or "dry_run — no changes").
```

After the subagent returns, surface its audit to the user verbatim. If the plan looks right, suggest re-invoking with `dry_run=false`.
