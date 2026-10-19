# ReMe MemLens Judge

Independent `answer_judge` Job for saved MemLens predictions. Install with
`pip install -e plugins/memlens-judge` and enable the `memlens-judge` plugin
in a separate evaluation application/workspace, not the candidate memory application.

Inputs are `question`, `reference`, `prediction`, and `category`, with optional
`subtype` and `old_answer`. The result is available as
`response.metadata.llm_judge_score` and `acc`. Metadata includes the raw response,
task rubric, native answer-preprocessing diagnostics and judge token usage.
Judge usage is evaluation cost, not method cost.

The official category/subtype rubrics, outdated-answer handling, verdict parser,
long-answer preprocessing and over-500-word zero rule are preserved. A successful
but unparseable judge output receives the official zero score; API errors instead
remain failed jobs for resumable judging of the same saved prediction.

The default judge is `qwen3.8-max`, with temperature 0 and thinking disabled.
Set `JUDGE_MODEL_NAME` to override the judge model independently of the candidate.
The benchmark's `settings.json` records the effective judge model, generation
parameters and protocol from the validated application configuration, after plugin
defaults and environment expansion. The snapshot excludes credentials and endpoints.
The output
budget is 256 tokens from the released OpenAI-compatible API judge branch
(`max_completion_tokens`), not the separate vLLM branch's 2,048-token setting.
These judge settings are independent of the candidate reader's thinking mode.

Source: [official MemLens judge](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/llm_judge.py).
Verbatim prompt strings and their MIT notice are packaged in `_memlens_judge_prompts.json`;
see also `LICENSE-CODE`. The plugin is independent of candidate code and other benchmarks.
