# MobileMem plugin

The benchmark configuration uses a local AgentScope wrapper that preserves the
pre-AgentReply evaluation protocol: an exact `exceed_max_iters` termination with
native `error=None` is accepted for memory and answers, including an empty result.
The original message stays in `raw.message`; the native failed status/error are saved
in `raw.benchmark_compat`.
Other termination reasons and native errors retain the core behavior. This exception
does not apply to calls with `output_schema`, which these benchmark paths do not use.
It adds no iterations or model calls and leaves the generic AgentScope backend unchanged.
The configuration selects a plugin-specific backend alias; its instance retains the
native `agentscope` backend identity required by Auto Memory's image capability check.
Candidate metadata forwards the compatibility marker. Auto Memory's session Response
does not forward the complete AgentReply; its native Agent state and successful session
receipts must be archived with the source revision to audit memory compatibility.

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
