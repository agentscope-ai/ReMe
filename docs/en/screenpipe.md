# Import reviewed Screenpipe notes

[Screenpipe](https://github.com/screenpipe/screenpipe) records desktop activity and
meeting context. ReMe can keep a selected, reviewed note from that history as a
resource, interpret it into a daily card, and retrieve it from later agent sessions.
This is a manual resource-import recipe using existing ReMe jobs.

## Prepare a source note

Choose one completed meeting or work episode in Screenpipe. Review the saved
notes, correct errors, and copy only the information you want ReMe to retain to a
UTF-8 Markdown file outside the ReMe workspace first. Include:

- A title and the recording time, including its timezone.
- The Screenpipe meeting ID or another source reference you can resolve locally.
- The reviewed note and any uncertainty that remains.

For example, a synthetic note might read:

```markdown
# Example project handoff

Source: Screenpipe meeting 42 on my-work-laptop
Recorded: 2026-01-15T10:00:00Z
Reviewed: yes

The team chose the CSV export for the handoff. The delivery date was not agreed.
```

Keep the source device label stable: meeting IDs can overlap across installations.
A note describing a workflow does not prove that every action succeeded.

## Add it to the workspace

Finish the [quick start](./quick_start.md) and identify the workspace used by your
running ReMe service. Copy the reviewed file into that workspace, for example:

```text
resource/2026-01-15/screenpipe-my-work-laptop-meeting-42.md
```

Use the meeting date in the directory name. Do not overwrite an unrelated file.
Only copy after review: an enabled resource watcher can process a new file as soon
as it appears. Processing may send the note to ReMe's configured model provider.

If the resource watcher is disabled, invoke the existing job against the same
workspace explicitly:

```bash
reme auto_resource workspace_dir=/absolute/path/to/your/workspace changes='[{"path":"resource/2026-01-15/screenpipe-my-work-laptop-meeting-42.md","change":"added"}]'
```

Replace the example path with your actual resource path. Use either the watcher
or the explicit call for initial ingestion, not both at once.

## Verify before relying on memory

Check the daily card under `daily/2026-01-15/` and confirm that its
`source_resource` points to the copied Markdown file. Compare the interpreted
card with the original note. Then use ReMe's [search and read interfaces](./memory_search.md)
to retrieve the handoff decision from another session, keeping the source path
with the answer. A file copy alone is not proof that interpretation or indexing
succeeded. Inspect errors before retrying a failed job.

## Corrections and removal

Keep the same resource path when correcting the note. The watcher can process the
edit; with the watcher disabled, call `auto_resource` with `change="modified"`.
The existing resource contract uses the exact `source_resource` link to update
the corresponding card.

Deleting the Screenpipe original does not delete the imported copy. To remove it
from ReMe, remove the selected resource and let the enabled watcher handle its
deletion, or explicitly submit its path with `change="deleted"`. Review any
later digest knowledge separately; deleting a source card is not a claim that
all derived knowledge has been erased.

See [Auto Resource](./auto_resource.md) for processing, retry and deletion semantics.
