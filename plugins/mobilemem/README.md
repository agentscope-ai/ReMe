# MobileMem plugin

Install this plugin from the repository:

```bash
pip install -e plugins/mobilemem
```

Enable the `mobilemem` ReMe plugin for the MobileMem-Omni benchmark. Its Steps use native ReMe Auto Memory,
source-line citations, BM25 retrieval, workspace containment, and an AgentScope bridge for the native image
reader. The plugin installs independently of MemLens. Evaluation uses the separate
[`mobilemem-judge`](../mobilemem-judge/README.md) plugin.

The runner supplies only original dialogue messages and image attachments. Memory extraction retains dates,
updates, personal relationships, app details and visual evidence; each question uses a fresh answer Agent. Source
session reading and `read_image` remain scoped to that user's workspace. Supplied captions and benchmark gold
annotations must never be passed to either Job.
