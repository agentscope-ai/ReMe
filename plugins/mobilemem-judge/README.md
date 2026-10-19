# ReMe MobileMem Judge

Independent `answer_judge` Job for saved MobileMem-Omni predictions. Install with
`pip install -e plugins/mobilemem-judge` and enable `mobilemem-judge` in a separate
evaluation application/workspace, not the candidate memory application.

Inputs are `question`, `reference`, `prediction`, and optional `evidence` (a list
of original evidence values). `judge_protocol` defaults to `membase_default`, the
executable MemBase rubric used by this benchmark. `native_strict` selects the
separately published strict rubric. Their protocol identifiers remain distinct.

Both modes retain the official input interpolation, WRONG-first verdict parsing,
and default-zero handling of successful responses without a valid label.
`response.metadata` contains `llm_judge_score`, `label`, `parse_default_zero`,
`default_judge_protocol`, `prompt`, `raw_response`, and `usage`. Provider errors
remain failed Jobs. Judge usage is evaluation cost, not method cost.

Defaults are `qwen3.8-max`, temperature 0, thinking disabled and 4,096 output tokens.
`JUDGE_MODEL_NAME` can override the Judge model independently of the candidate.
Candidate methods and their workspaces are not dependencies of this plugin.
The benchmark runner invokes this Job in a separate application and saves its
prompt, provider output and verdict alongside the candidate prediction. Its
`judge-settings.json` records the validated model, generation/retry settings and
protocol identifier through a credential-free whitelist, rather than recording
only the default model name.

Sources: [MemBase executable judge](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/inference_utils/prompts.py)
and [MobileMem published strict rubric](https://github.com/zjunlp/MobileMem/blob/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval/question_answering_and_judge_prompts.txt).
The upstream prompt/parser code retains its MIT notice in `LICENSE-CODE`.
