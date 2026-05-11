---
description: Distill the current session's active events into the topic graph via the reme-distiller subagent.
argument-hint: [optional event-folder paths or hint]
---

Spawn the `reme-distiller` subagent to handle cold-path R-M-W on the current session's events. The subagent runs in its own context window and returns a short audit summary; the main session stays focused on the user's task.

Use the Agent tool with `subagent_type: reme-distiller` and the following prompt:

```
End-of-task distillation pass.

Hint: $ARGUMENTS

If the hint above doesn't list specific event folder paths, discover them yourself:
- memory_list metadata={role: observation, status: active} filtered to today's date prefix.
- For each candidate, memory_get the index, follow ## Materials, decide whether the content has already been distilled (skip if so).

Apply the R-M-W decision rules (SKIP / CONTRADICT / EXTEND / CREATE / STATUS-FLIP) per the protocol you have loaded. Mutations go through memory_create / memory_update / memory_property_update — every claim-role topic needs `confidence`, every distilled event ends with `status=distilled`.

Return a concise audit: applied ops with paths, skipped events with reason, failed ops with reason.
```

After the subagent returns, surface its audit to the user verbatim — don't paraphrase.
