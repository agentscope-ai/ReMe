# Context compression: Jev across two summary strategies

**English** | [简体中文](./README_ZH.md)

Summaries can omit identifiers, configuration values or log details needed for later answers. This experiment asks whether
appending a few Jev-selected original evidence blocks improves information retention across two summary strategies,
while keeping mean input shorter than the full history.

This is a **synthetic-task comparison from 2026-10-16**. Hermes collects trajectories; evaluation calls Qwen directly.
There is no resumed agent/tool loop, archive recovery or task-completion measurement. Production `compact_context` is separate.

## Experiment design

### Data and cohorts

There are **14 predefined query/answer templates**: one reused original payment trajectory and thirteen newly collected
read-only Hermes trajectories over synthetic upstream fixtures. For collection, queries asking for file changes or cron jobs
are adapted to evidence investigation and proposed actions. Source workspaces are immutable; only allowlisted `read_file`
is exposed. The appendix lists all queries and expected answers.

Fixtures come from MIT-licensed Hermes code, not private maintainer production trajectories. Long logs and build outputs
use deterministic original head/matching-line views, not every byte of the upstream workspace; no noise or renamed identifiers
are added. Cohorts are fixed before collection: evidence-dependent tasks with at least 2,000 source-snapshot characters
enter the primary cohort, with the original payment source pinned separately.

| Cohort | Tasks |
| --- | --- |
| Primary (4) | `payment`, `checkout_504`, `server_error`, `build_token` |
| Short/query-only controls (10) | `backup`, `refactor`, `three_files`, `active_config`, `dependency_version`, `retry_backoff`, `meeting_notes`, `targeted_edit`, `package_versions`, `checksum_anomaly` |

### Methods

There are **13 methods**: Full context, plus S and H each with Jev Top 0–5. Top 0 means summary only.

| Method | Input construction |
| --- | --- |
| Full context | Original system/task and public history before the final collection report, including tool calls/results |
| Generic summary S | Qwen handoff summary of all evidence units and the original task, using the prompt in [`cases.py`](./cases.py) |
| Hermes compaction summary H (small-budget setting) | The same Qwen model through Hermes' native summary-generation pipeline, summarizing evidence history with the original task as focus |
| S/H + Jev Top N | The entire corresponding summary body, followed by the top N whole original evidence units |

Each S/H stays byte-identical across its own Top 0–5 conditions. Both mechanisms share exactly the same Jev ranking and
selected originals for each task/generation/N. S receives structured evidence units; H receives public history including
tool calls/results, so their differences extend beyond the prompt. Compressed arms use the same
`Recorded evidence:` / `Summary:` / `Original evidence:` framing, preserve the original system/task and append the same exam.
Internal metadata, thinking and final collection reports are excluded. Exams and gold never enter collection, summary or decision requests.

H calls `agent.context_compressor.ContextCompressor._generate_summary` with `tail_mode=legacy`, reusing native
serialization, structured checkpoint templates, redaction and provenance validation. The experiment overrides the summary
budget and adds a total word limit, retaining Historical Task Snapshot/no-user sentinel and Critical Context while allowing
other sections to be omitted. Hermes' helper removes the control prefix without rewriting facts; lean-mode extra session
logs and appendices are not used. H is **Hermes compaction summary under a small-budget setting**: it invokes generation
directly rather than full `compress()`. Automatic triggering, head/tail selection and post-compression conversation repair
are outside this comparison. H-only inputs must actually shorten Full; original-history fallback is not accepted.

### How Jev selects originals

Evidence units are **chunks of tool-result text**, with an ID, file path extracted from tool arguments, source positions
and body text. Long results are split into related events or text segments. Top N does not mean N complete tool calls:
units do not carry the full tool name and all call arguments. See `event_units` in [`cases.py`](./cases.py) and
`decision_input` in [`run.py`](./run.py).

Jev receives the original task and `evidence_units` (`id`, `file`, `text`, **including tool-result bodies**). It answers a
binary probability question for each unit: does it contain material information needed to continue the task correctly?
Ranking is not conditioned on the summary or exam. Requests contain at most ten units, an experiment batch size rather
than a fixed service limit. The three ranking generations saved 54 Jev requests: thirteen tasks need one request each per
generation, while the 41-unit `server_error` needs five (10/10/10/10/1). Completed rankings are reused across both summaries
and all N values; Top N does not determine request count.

Select a strict descending-probability prefix, break ties by ID, then restore source order. There is no threshold, skipping
or partial-unit truncation. The actual count is `min(N, available units)`, saved as `actual_n`. The original payment source
has ten units (maximum Top 10); this study sweeps Top 1–5 throughout.

### Budgets, repetitions and scoring

- Three independently requested summary/ranking generations per task, two answers per context:
  **14 × 3 × 13 = 546 contexts, 1,092 answers**. Each method has 24 primary and 60 control answers; repetitions are not independent tasks.
- Summary-body allowance: `max(64, min(256, int(0.2 * evidence_tokens)))`. S allows up to three total length-check attempts,
  with word target `max(12, int(allowance * 0.4 * 0.8**attempt))`.
- H also allows three total length-check attempts. Requested budget:
  `max(12, int(allowance * 0.8**attempt))`; word limit: `max(18, int(requested_budget * 0.35))`.
  Length adjustment never uses answer feedback.
- Before answering, compute the complete-input ceiling from protected prompts, the five largest whole units, summary
  allowance and 96 reserve/framing tokens; enlarge it first if serialized S needs more space. This is an incremental
  comparison, **not equal-length or a universal 1,000/2,000-token ceiling**, and not every Full exceeds its ceiling.
  See [`suite_run.py`](./suite_run.py).

| Case | Evidence units | Summary allowance | Complete-input ceiling |
| --- | ---: | ---: | ---: |
| payment | 10 | 256 | 4,496 |
| checkout_504 | 5 | 256 | 5,246 |
| server_error | 41 | 256 | 3,185 |
| build_token | 8 | 256 | 3,859 |

Answers must return `{"facts": {...}}`. Predefined fields require recursive exact JSON types/values: integers differ from
booleans and floats; strings and ordered arrays must match exactly. Only payment `no_retry_http_statuses` ignores array order.
Missing/malformed fields and failed planned conditions score zero and remain in the denominator. No answer aliases,
rewrites or retries are added after scoring; **no LLM judge is used**. Average fields within answers, answers within
generations, generations within tasks, then tasks equally. Primary/control cohorts are separate. All-fields-correct rates
are also saved. See [`suite.py`](./suite.py).

Charted answer input uses actual Qwen API `prompt_tokens`, including system/task, history and exam, excluding summary
and Jev requests. Budget checks use a local tokenizer proxy; these are distinct measures.

## Results and conclusions

### Four primary tasks

![Answer quality and downstream input across two summary strategies](../../docs/figure/compaction-topn-results.svg)

| Method | Fact score | Qwen answer input tokens |
| --- | ---: | ---: |
| Full context | 100.0% | 4,014 |
| S | 82.0% | 2,480 |
| S + Jev Top 1 | 83.0% | 2,753 |
| S + Jev Top 2 | 94.5% | 2,974 |
| S + Jev Top 3 | 99.5% | 3,241 |
| S + Jev Top 4 | 98.1% | 3,496 |
| S + Jev Top 5 | 97.4% | 3,660 |
| H (Hermes, small budget) | 79.4% | 2,432 |
| H + Jev Top 1 | 85.3% | 2,705 |
| H + Jev Top 2 | 92.9% | 2,927 |
| H + Jev Top 3 | 97.9% | 3,193 |
| H + Jev Top 4 | 96.9% | 3,448 |
| H + Jev Top 5 | 96.9% | 3,613 |

On these four tasks, Top 3 raises S from **82.0% to 99.5%** and H from **79.4% to 97.9%**, reducing mean answer input by
**19.3% / 20.4%** relative to Full. A few originals can therefore supplement exact evidence lost by either summary.
Top 4–5 do not improve the point estimates; retaining more original text does not guarantee greater accuracy.

Every condition fits its input ceiling; all 42 H-only contexts shorten Full. However, at Top 3–5 only **9/12 primary contexts
per method** are shorter than their paired Full. Across H fusion arms, 172/210 contexts shorten Full (51/60 primary,
121/150 controls); the rest remain scored. **Mean savings do not imply per-case savings.**

This is an information-retention diagnostic on a small synthetic panel. Payment was used during earlier development,
not held out independently. There is no equal-length longer-summary or alternative-selector control, so the gains cannot
all be attributed to Jev ranking. The results establish neither a generally optimal N, statistical equivalence to Full,
nor improved agent task-completion rates.

<details>
<summary>Ten short/query-only controls</summary>

| Method | Fact score | Qwen answer input tokens |
| --- | ---: | ---: |
| Full context | 100.0% | 2,727 |
| S | 67.1% | 2,315 |
| S + Jev Top 1 | 84.7% | 2,493 |
| S + Jev Top 2 | 87.5% | 2,573 |
| S + Jev Top 3 | 92.9% | 2,687 |
| S + Jev Top 4 | 92.9% | 2,722 |
| S + Jev Top 5 | 92.5% | 2,722 |
| H (Hermes, small budget) | 73.2% | 2,308 |
| H + Jev Top 1 | 80.1% | 2,486 |
| H + Jev Top 2 | 92.5% | 2,566 |
| H + Jev Top 3 | 99.2% | 2,680 |
| H + Jev Top 4 | 99.2% | 2,715 |
| H + Jev Top 5 | 98.8% | 2,715 |

These are not ten additional long-context tasks. Top N saturates when fewer source units are available.

</details>

## Related Hermes Agent experiment

[Hermes Agent PR #116246](https://github.com/NousResearch/hermes-agent/pull/116246) reports a different experiment:
`fast-jev-compaction` replaces summaries with Jev-scored tool-call/result retention while keeping user/assistant text
verbatim. On three real 500K-token lineage prefixes, with 15 recall questions each, the authors report:

| Method | Mean recall | Retained tokens |
| --- | ---: | ---: |
| Summary only | 43.3% | 55K |
| Summary + one `session_search` recovery round | 78.9% | 55K |
| Jev, plugin defaults | 75.5% | 115K |
| Jev-ranked / recency-ranked, same 60K tool-evidence budget | 77.8% / 77.8% | 176K / 177K |

At the default threshold, all 851 candidate tool calls were dropped; the closed-book gain over summary-only therefore
came from retaining user/assistant text, not selective tool retention. Repeated-compaction simulations also found that
the growing text floor eventually prevents further savings. The authors recommend against adopting that strategy as
Hermes' primary compaction mechanism; the merged PR adds evaluation, not production adoption.

Our experiment instead keeps a summary and appends selected original evidence, with tool-result bodies visible to Jev.
It uses small synthetic tasks, a small-budget summary setting and exact-field scoring, without archive recovery or
continuous compaction. Its scores cannot be compared directly with the Hermes recall percentages, and it does not
establish superiority over Hermes' production summary-plus-recovery path. The experiments address different strategies;
our missing equal-budget selector controls also prevent attributing the gains solely to Jev ranking.

## Reproduction

### Models and runtime

| Component | Version/model |
| --- | --- |
| Hermes Agent `AIAgent` | commit `30463589eb54fa0f48724594cc0e98e6b995729b` |
| Collection, S/H generation, answering model | `qwen3.8-max` |
| Jev (through AgentScope/TypeSafe) | Resolved response version `jev-1.13.0` |
| Experiment implementation snapshot | ReMe commit `50c2c806446c7ae6acc66217afd6e5edc2114046` |

Qwen requests/responses identify only a model alias, not an immutable provider weight revision. Service updates and
nondeterminism may change results even at temperature zero. The run used the public China (Beijing) DashScope
OpenAI-compatible Chat Completions endpoint; use the same `LLM_BASE_URL` service for the closest rerun.

<details>
<summary>Dependency versions and request parameters</summary>

Two separate Python environments were used. These versions were **verified from the retained environments after the
run**, not from a pre-run lockfile; they are a reproduction reference rather than proof of a fully locked installation.
The local run used Python **3.13.3**, macOS arm64. No local GPU performs inference.

| Package | ReMe evaluation environment | Hermes collection/summary environment |
| --- | --- | --- |
| `openai` | 2.53.0 | 3.31.0 |
| `agentscope` | 2.0.10 | — |
| `typesafe-sdk` | 0.7.5 | — |
| `httpx` | 0.28.1 | 0.28.1 |
| `httpx2` | — | 2.13.1 |
| `python-dotenv` | 1.2.2 | 1.2.4 |
| `pydantic` | 2.13.4 | 2.14.1 |
| `tokenizers` | 0.23.3 | 0.23.3 |

`tokenizers` was imported from a separate package directory on `PYTHONPATH`; it was not installed in either venv.
A new setup may install that exact version into each venv. The local artifact bundle includes both environments'
distribution metadata, recorded after the experiment. It does not contain credentials or virtualenv binaries.

Token accounting uses `Qwen/Qwen3.5-9B`'s **`tokenizer.json`**, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, with SHA-256
`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.
This is a **budget proxy**, not a claim that Qwen 3.8 uses that tokenizer. Reported downstream inputs use actual API usage.

#### Request parameters

| Stage | Sampling/output parameters | Execution limits |
| --- | --- | --- |
| Hermes collection | `max_tokens=1024`; `reasoning_effort=none`, `enable_thinking=false`; no explicit temperature or seed | 12 agent iterations, 240-second run budget; four workers; 360-second subprocess timeout |
| S generation | `temperature=0`, `max_tokens=4096`; thinking disabled; no seed | OpenAI client: 90-second timeout, `max_retries=1` |
| H generation | `temperature=0`; no explicit API `max_tokens`; thinking disabled; no seed | Same 90-second client timeout / one SDK retry; four workers; 600-second subprocess timeout |
| Jev ranking | Binary probability question per evidence unit; batch size 10 | Adapter timeout 45 seconds; `max_retries=2`, retry delay 0.5 seconds |
| Answering | `temperature=0`, `max_tokens=2400`, `response_format={"type":"json_object"}`; thinking disabled; no seed | 90-second client timeout, `max_retries=1`; four concurrent conditions, two sequential answers each |

For direct Qwen calls, thinking is disabled with `reasoning_effort="none"` and
`extra_body={"enable_thinking":false}`. The collection wire request carries `enable_thinking=false` at top level.
H's API output limit is intentionally not its summary-body allowance: acceptance uses the local body-token check.
Captured request bodies are authoritative for the recorded run.

Collection exposes only allowlisted `read_file`; source workspaces are read-only. Automatic compression and micro
compaction are disabled during collection. Memory/context files and background review are skipped. Each worker has an
isolated `HERMES_HOME`, workspace and session DB. See [`hermes_worker.py`](./hermes_worker.py).

</details>

### Fresh collection and evaluation

Use ReMe's development environment and a separate Hermes Python environment for collection/H generation; both must
import `tokenizers`. Set `LLM_API_KEY`, `LLM_BASE_URL`, `DECISION_MODEL_API_KEY`, optionally `DECISION_MODEL_BASE_URL`,
in `.env`; request `DECISION_MODEL_NAME=jev-1.13.0` when the service supports it. Every command requires explicit `--live`
and new output directories outside Git; existing runs are never overwritten.

```bash
# Freeze all task/answer definitions, then collect 13 new read-only trajectories.
# Reuse the pinned original payment trajectory as the fourteenth source.
python -m benchmark.compaction.suite_collect --live \
  --hermes /path/to/hermes-agent \
  --hermes-python /path/to/hermes-environment/bin/python \
  --payment-trajectory /path/to/original/trajectory.full.json \
  --output /path/outside/repository/corpus

# Freeze S and Jev rankings, then run the seven-method Top-N sweep.
python -m benchmark.compaction.suite_run --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/evaluation

# Generate Hermes compaction summaries under the small-budget setting.
# An existing --native directory reuses frozen generations only after source validation.
python -m benchmark.compaction.suite_native --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --evaluation /path/outside/repository/evaluation \
  --native /path/outside/repository/native-generations \
  --hermes /path/to/hermes-agent \
  --hermes-python /path/to/hermes-environment/bin/python \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/combined-results

# Reuse all frozen summaries/rankings and 672 answers; add only H + Top 1–5.
python -m benchmark.compaction.suite_fusion --live \
  --corpus /path/outside/repository/corpus/corpus.json \
  --evaluation /path/outside/repository/evaluation \
  --combined /path/outside/repository/combined-results \
  --native /path/outside/repository/native-generations \
  --tokenizer /path/to/cached/qwen/tokenizer.json \
  --output /path/outside/repository/dual-summary-results
```

This reproduces the protocol; new agent calls may produce different trajectories. `suite_fusion` re-answers only
H + Top 1–5, retaining the other eight arms, rather than re-answering all thirteen. To reuse the recorded H generations,
also pass `--native-evaluation /path/to/pre-amendment/evaluation` to either `suite_native` or `suite_fusion` when reusing
its cache. Both runners require `generation/validation.json` to match the SHA-256 of the source file used for generation
and all three summary repeats to be present. By default that source is the current evaluation's `source.json`;
`--native-evaluation` explicitly identifies the original source. Only the held-out `exam` may differ from the current
source; the summary generator never receives that field. Missing provenance, a hash mismatch or changed evidence is
rejected before answering calls. Preserve rejected caches and use a new `--native` path to generate replacements.

### Frozen offline verification

Public Git contains the experiment code, fixture definitions, rules and curated results. The original payment trajectory,
frozen corpus and wire records are **not committed**. Exact replay requires the local artifact bundle; it is not currently
a public one-command reproduction. Do not substitute another payment trajectory: its expected SHA-256 is
`0f37e6510ffafd920d088452d8719026b00fd4e74f87f7dd1a42a85c8287e6f1`.

The prepared bundle is outside Git:
`/Users/yuli/workspace/trajectory-runs/compaction-dual-summary-reproduction-20261016.tar.gz`.
Its SHA-256 is `40ee93c9117dfe59cf0b4dc8887ba4b2c62fa16f3bc97f2dac6d536343f942a3`.
It contains 3,187 files, a relative-path SHA-256 manifest, environment metadata and `verify.py`.
With ReMe's development environment installed, from the ReMe checkout:

```bash
# Extract outside the repository. Substitute the directory holding the archive.
export REPRO_DIR=/path/outside/repository
shasum -a 256 "$REPRO_DIR/compaction-dual-summary-reproduction-20261016.tar.gz"
tar -xzf "$REPRO_DIR/compaction-dual-summary-reproduction-20261016.tar.gz" -C "$REPRO_DIR"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python \
  "$REPRO_DIR/compaction-dual-summary-reproduction-20261016/verify.py"
```

Expected result: **3,187 artifact hashes, 546 contexts and 1,092 grades verified; zero API calls**.
The bundle's `corpus/`, `evaluation/`, `native/`, `native-source/`, `combined/` and `final/` directories correspond to
the run stages above. Pass `--native-evaluation <bundle>/native-source` when reusing its H generations.

## Appendix

<a id="queries-and-expected-answers"></a>

<details>
<summary>Queries and expected answers</summary>

Each initial query supplies the collection goal; its follow-up asks for the JSON fields below. One task and its follow-up
are one sample. These are the original English queries; the Chinese version paraphrases them. Actual read-only collection
adaptations and field descriptions are defined in [`suite.py`](./suite.py) and frozen requests. Gold keys, types and strings
are preserved exactly.

| ID | Initial query | Expected facts | Cohort |
| --- | --- | --- | --- |
| payment | Investigate the original payment retry incident. | `{"incident_id": "PAY-4827", "invoice_id": "INV-731", "successful_charge_id": "CH-991", "amount_minor": 12900, "attempt_keys": ["pay-a", "pay-b"], "first_attempt_committed": true, "max_attempts": 3, "no_retry_http_statuses": [402, 422], "key_generation_line": 5, "regression_attempt_counts": {"timeout_then_success": 2, "validation_422": 1, "three_timeouts": 3}}` | Primary |
| backup | A while back I told you one of my repos needed nightly backups — find which repo that was in our past conversations (do not guess). Then: write backup.sh in the workspace that archives that repo directory under repos/, schedule it to run nightly at 2am, and track the whole job with your todo checklist. Report back which repo it was and what you set up. | `{"repo": "sparks-data", "ignored_repo": "toybox", "data_text": "important"}` | Control: short_fixture |
| refactor | In src/, every requests call uses timeout=30. Bump them all to timeout=45, add a CHANGELOG.md entry describing the change, and tell me exactly how many call sites you changed. | `{"old_timeout": 30, "call_sites": 3, "functions": ["fetch_a", "fetch_b", "fetch_c"], "methods": ["get", "post", "head"], "old_version": "1.0"}` | Control: short_fixture |
| three_files | This is a multi-step job — track it with your todo checklist tool and keep it updated as you go, marking each step completed when verified. Steps: (1) create greet.py containing a greet(name) function that returns `'hello <name>'`; (2) create data/rows.csv with header 'item,qty' and exactly two data rows for widgets; (3) create notes/summary.md stating how many files you created. Finish only when all three are done and checked off. | `{"file_count": 3, "header": "item,qty", "data_rows": 2}` | Control: query_only |
| active_config | Search this workspace for wherever the retry_limit setting is configured and tell me its current value. | `{"retry_limit": 7, "backoff": 2, "active_file": "conf/app.ini", "old_file": "conf/old.ini"}` | Control: short_fixture |
| checkout_504 | Investigate the checkout 504 incident: distinguish the failed rollback, root cause and final database fix. | `{"setting": "statement_timeout", "value_seconds": 45, "index_mode": "CONCURRENTLY", "root_cause": "CREATE INDEX without CONCURRENTLY", "rolled_back_version": "v2.30.2", "rollback_fixed": false, "deployed_version": "v2.31.0", "initial_error_percent": 8}` | Primary |
| dependency_version | What exact left-pad version is pinned, and which file declares the test script? Record the package scripts for a handoff. | `{"version": "1.3.0", "declaration": "package.json", "test": "vitest run", "build": "tsc -p .", "package_version": "4.2.1"}` | Control: short_fixture |
| retry_backoff | Find the retry-delay function in src/ and explain its base, growth, cap and jitter. | `{"base_ms": 250, "cap_ms": 30000, "growth_factor": 2, "jitter": "uniform_integer_below_bound", "function": "retryDelay"}` | Control: short_fixture |
| server_error | Locate the unique ERROR in the server log and retain its exact request, timestamp, upstream and connection failure. | `{"request_id": "req-7f3d9", "timestamp": "2026-08-08T23:41:17Z", "status": 502, "upstream": "payments", "error": "ECONNREFUSED", "address": "10.0.4.17:8443"}` | Primary |
| meeting_notes | Read the meeting notes and report the third bullet exactly, retaining the other agreed actions. | `{"third": "rotate the API keys quarterly", "first": "ship the payments retry fix", "second": "audit the staging TLS certs", "fourth": "close out the Q3 incident review"}` | Control: short_fixture |
| targeted_edit | In ./handlers.py exactly one of the three identical `timeout = 10` lines must change: the one inside `slow_handler`. Change it to `timeout = 60` using the patch tool. Then run `python3 check_handlers.py` from . and reply with its output. | `{"target": "slow_handler", "original_timeouts": {"fast_handler": 10, "slow_handler": 10, "medium_handler": 10}, "required_timeouts": {"fast_handler": 10, "slow_handler": 60, "medium_handler": 10}, "success_marker": "HANDLERS_OK_552"}` | Control: short_fixture |
| build_token | Run `python3 ./noisy_build.py` (it prints a lot). Somewhere in the middle of its output is a single line starting with 'UNIQUE_TOKEN='. Reply with the full token value. | `{"token": "tok_9f31c_middle", "step": 2000, "total_steps": 4000, "prefix": "UNIQUE_TOKEN="}` | Primary |
| package_versions | The project . has three package dirs: pkg_a, pkg_b, pkg_c, each containing version.txt. Working through the directories, collect the three versions and create ./versions.txt containing them comma-separated in order (a,b,c). Reply DONE plus the joined string. | `{"versions": {"pkg_a": "1.4.2", "pkg_b": "0.9.7", "pkg_c": "3.2.1"}, "joined": "1.4.2,0.9.7,3.2.1"}` | Control: short_fixture |
| checksum_anomaly | The file ./records.log contains exactly one line starting with 'ANOMALY:'. Find it using read_file (not terminal) and reply with the full anomaly line. | `{"line": "ANOMALY: checksum drift detected in shard 7 (code X99Q)", "code": "X99Q", "shard": 7, "line_number": 4201}` | Control: short_fixture |

</details>

<details>
<summary>Protocol amendments, audit and historical artifacts</summary>

The initial seven-arm, 588-answer run exposed two ambiguous format instructions: payment test counts were returned as an
array, and meeting bullets included list markers. Gold/grader stayed unchanged; only the payment named JSON mapping and
marker-free meeting text were clarified. Both tasks were re-answered across all seven arms with frozen S/rankings; the
initial 84 answers remain as excluded preflight records. The other twelve tasks retained 504 answers. H-only then added
84 answers, and H fusion added 420, forming the final results; all 42 H summaries, rankings and 672 prior answers were reused.
Cohorts, evidence and gold values were not adjusted based on outcomes.

The first forced H generation included a long control prefix; 40/42 inputs exceeded Full and remain integration preflight
records only. Body framing and word constraints were fixed before fresh scored answers, without changing S, gold, cohorts
or Jev rankings. H used pre-amendment source files; only the `exam` field, excluded from generation, differs.
The early payment `budget.json` records 4,459; the final question adds 37 tokens, making final `results.json`'s 4,496 the
scored ceiling. Frozen artifacts are preserved, not rewritten.

The offline audit checks source hashes, frozen S/H, identical nested Top-N originals, captured inputs, every grade,
input ceilings, unchanged prior results and configured-key exclusion. All 546 conditions are valid. Raw files, sessions,
API requests/responses, code snapshots and per-field grades remain outside Git. Historical directories under
`/Users/yuli/workspace/trajectory-runs/`:
`hermes-fixture14-20261016/` (corpus), `compaction-fixture14-20261016-v2/` (S/rankings),
`compaction-fixture14-hermes-adapted-20261016/` (H), `compaction-fixture14-final-20261016/` (original eight arms),
`compaction-fixture14-dual-summary-20261016/` (final thirteen arms).

</details>
