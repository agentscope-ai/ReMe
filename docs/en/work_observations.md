# Work observations

[`proactive.yaml`](../../reme/config/proactive.yaml) collects all displays and new local Codex dialogue every five
minutes. One `append_event(agent_id=work, events=[...])` call delivers each batch to the work Agent. Collection does
not call a model.

Start the service:

```bash
reme start config=proactive
```

Inspect the work Agent from another terminal:

```bash
reme manage_proactive_agent action=status agent_id=work
```

`work_observation_init` establishes a startup baseline; `work_observation_cron` uses `*/5 * * * *`. The existing
`screen_capture_cron` remains a disabled standalone collector to avoid duplicate captures. To disable work collection,
set both `work_observation_init.enabled` and `work_observation_cron.enabled` to `false`.

## Inputs and state

- The default read-only Codex source is `${CODEX_HOME:-~/.codex}/sessions`. Change `codex_home` in both Jobs' Step
  configurations to select another source. This is independent of ReMe's workspace-relative
  `codex_app_server.codex_home`.
- First activation skips existing complete records. Restarts retain cursors and catch up on changes made while offline.
- The current parser uses Codex rollout `event_msg/item_completed` records: actual `UserMessage` items and
  `AgentMessage.phase=final_answer`. Reasoning, tool output, commentary, raw injected context and ReMe's own sessions
  are excluded.
- Dialogue retains original timestamps, project `cwd`, thread, turn, message ID and text. Image attachments retain
  only their count, without copying images or temporary paths.
- Discovery, reads, screenshots and parsing run on worker threads. Cursors are byte offsets; incomplete lines wait
  for the next pass. `max_scan_bytes` defaults to 32 MiB per pass, allowing one complete record within the existing
  20 MiB file limit to cross the scan budget. Backlogs span multiple passes with rotating file order. This targets
  append-only logs and does not guarantee detection of arbitrary historical in-place edits.

Files remain inside the configured workspace:

```text
resource/screenshot/<date>/<batch>/metadata.json, 1.png, 2.png…
resource/codex_dialogue/<date>/<batch>/metadata.json, dialogue.jsonl
metadata/work_observations/work/baseline.json
metadata/work_observations/work/state.json
metadata/work_observations/work/history.jsonl
metadata/work_observations/work/pending/     # Only while a batch is incomplete
```

Complete batches are staged before publishing resources, delivering events and committing cursors. Failures retry
the original batch with the same event IDs, times and paths. A failed source does not prevent successful sources
from being delivered. To rebuild the cursor index, remove only `state.json`, retaining `baseline.json`, `history.jsonl`,
pending batches and resource files. Corrupt state fails explicitly instead of resetting cursors.

Screenshots require `mss` from `reme-ai[core]` and operating-system screen recording permission. The work Agent follows
its own `MEMORY.md`; unfinished dialogue requests do not establish completed work, and screenshots are historical
observations from their capture time.
