# Jev Context Agent

Context Agent observes task snapshots while tools execute and selects source-backed memory evidence that may help the next action. The host publishes events, leases ready hints before its next model request, and acknowledges successful insertion. The main agent keeps its existing memory search capability.

This is an optional embedded feature. The default configuration does not enable it. The `context_agent` configuration adds a private `evaluate_context` Job and a coordinator in shadow mode; it exposes no additional HTTP/MCP tools.

## Configuration and evaluation

Install the existing `reme-ai[classifier-jev]` extra and set `CLASSIFIER_API_KEY` in the environment or `.env`. Existing settings `CLASSIFIER_BACKEND`, `CLASSIFIER_MODEL_NAME`, and `CLASSIFIER_BASE_URL` default to `jev`, `jev-latest`, and `null`.

```python
from reme.application import Application
from reme.config import resolve_app_config
from reme.utils import load_env

load_env()
config = resolve_app_config(config="context_agent", log_config=False)
config["workspace_dir"] = "/absolute/path/to/your/workspace"

async def evaluate(event):
    async with Application(**config) as app:
        response = await app.run_job("evaluate_context", event=event)
        if not response.success:
            return None
        return response.answer
```

An event contains `session_id`, `task_id`, `event_id`, a session-wide monotonically increasing `revision`, `event_type=task_start|tool_start|tool_result`, and `task_snapshot`. The snapshot requires `goal`; optional fields include `recent_context`, `current_tool`, `tool_result_excerpt`, `query_terms`, and delivered candidate IDs in `seen_evidence`. See `reme/schema/context_agent.py` for the public models.

Build the index through the existing indexing workflow first. Evaluation discovers candidates from the index, then verifies and reads evidence from permitted workspace files. Missing or stale indexes may yield no candidates. The complete result is in `Response.answer`: a successful decision is `silent` or `hint`, with excerpts, paths, line ranges, and source versions. Timeout, invalid answers, and provider failures return an unsuccessful Response; callers must check `success`.

## Host coordination

Reuse the same Application and coordinator across events. Change `components.context_agent.default.limits.mode` from `shadow` to `hint` to allow delivery.

```python
agent = app.context.components["context_agent"]["default"]
await agent.submit(event, allowed_paths=["daily", "digest"])

# Continue tool execution. At a serialized checkpoint before the next model call:
hint = await agent.take_ready(
    event["session_id"], event["task_id"], event["revision"],
    allowed_paths=current_allowed_paths,
)
if hint is not None:
    # Deduplicate by hint_id and insert hint.text as evidence, not system instructions.
    # Acknowledge only after successful insertion.
    await agent.ack(event["session_id"], hint.hint_id)
```

`submit` does not wait for retrieval or classification. Each session has one evaluation task and one latest pending snapshot; dense events are coalesced. `take_ready` never waits for unfinished model inference, but reads source files to recheck versions and permissions. The host must supply its current permission scope and keep delivery serialized with task updates. Old revisions and expired evidence are rejected. Revisions must keep increasing when the task ID changes within a session.

Evidence becomes seen only after `ack`. A short lease permits retry after a failed insertion or lost acknowledgement; the host must deduplicate by hint ID. Call `dispose_session(session_id)` when the session ends. Application shutdown cancels and awaits outstanding coordinator work. `last_result(session_id)` supports shadow inspection; `wait_idle()` is for offline replay, not production checkpoints.

## Budgets and boundaries

Defaults are an 800 ms total evaluation deadline, up to eight candidates, and up to two evidence items. The classifier has no retries. Each current session task permits ten evaluations and three delivered hints, with a 30-second hint cooldown. Application limits are 128 sessions and four concurrent evaluations. Configure evaluation `options` and coordinator `limits` separately; `limits.enabled=false` disables event submission.

Request fitting uses UTF-8 byte length with an envelope margin as a conservative token estimate, rather than an exact Jev tokenizer or billing count. It budgets task state, evidence, and questions together, removes whole candidates if necessary, and fails if one candidate cannot fit. Results include fitting counts, provider usage, and stage timing.

This feature reads memory without changing source files or cancelling tools or main-model requests. Hosts need event and context-insertion capabilities; exposing an MCP tool alone does not provide proactive delivery. The coordinator directly depends on retrieval and classification components, so the existing Application dependency lifecycle closes its tasks before those dependencies.

## Real Jev integration cases

Explicitly run against repository `.env` using synthetic workspaces:

```bash
REME_RUN_JEV_INTEGRATION=1 python -m pytest tests/integration/test_context_agent_jev.py -v -s
```

| Case | Distractors | Expected assistance |
| --- | --- | --- |
| Production connectivity incident | Obsolete direct-route workaround, staging instructions, known certificate facts, unrelated preference | Find current CONNECT gateway and split-DNS evidence; avoid incorrect IP pinning and restarts; stay silent after resolution |
| CFO month-close analysis | Expired order metric, marketing bookings, known schemas, unrelated formatting preference | Find net-revenue/refund contract and Shanghai month boundaries; change the scripted calculation from bookings 600 to recognized net 250 |

Both cases use real local BM25 retrieval and real Jev, check delivery, and verify source files remain unchanged. The main-agent host is scripted: these tests validate evidence selection and delivery, not the task success rate of a separate generative main model. Reports contain only synthetic task data and statistics.

Integration cases allow 20 seconds for classification and evaluation to test decision quality independently of the fast-path budget. The first validation measured approximately 4.58 seconds and 0.66 seconds, including first-call initialization effects. These observations do not establish a 500 ms p95 target. The default 800 ms deadline may suppress hints; measure cold starts, tail latency, and task benefit separately in the deployment environment.
