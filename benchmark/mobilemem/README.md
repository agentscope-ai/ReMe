# MobileMem-Omni Visual Reasoning

Evaluate ReMe Image on the published `visual_reasoning` questions using each user's complete chronological
dialogue history. Original dialogue text, including any descriptions already in it, is retained alongside images.
Separate summaries/captions, reference answers, evidence annotations and question types are not added to candidate Jobs.
Native `multi_visual_reasoning` questions are not relabelled or included.

## Run

Install both plugins and extract the official archive without changing its images:

```bash
pip install -e ".[as]" -e plugins/mobilemem -e plugins/mobilemem-judge
mkdir -p /path/to/omni/image
python -m zipfile -e /path/to/omni/image.zip /path/to/omni/image
export LLM_API_KEY="..."
export LLM_BASE_URL="..."
python -m benchmark.mobilemem.run --dataset /path/to/omni --output /path/to/exp-xxxx
```

[config.yaml](config.yaml) supplies defaults: `qwen3.8-max`, four concurrent users and 16 concurrent questions.
Override them with `--model`, `--user-concurrency` and `--query-concurrency`.
Every run creates `<output>/mobilemem/<model>` and rejects an existing directory.

## Evaluation

Each user has a fresh workspace. All source sessions are ingested chronologically through multimodal
`auto_memory`, then indexed once. Every question starts a fresh 30-iteration `agentic_answer` Agent with
BM25 `search`, `read` and `read_image`. Search covers daily notes, digest notes and original session text.
Original Image IDs accompany source images; saved native attachments can be reopened by the answer Agent.

The [candidate preset](../../plugins/mobilemem/src/reme_mobilemem/benchmark.yaml) uses the same model for memory,
answering and compression: temperature 0, thinking enabled, 65,536 output tokens, 1,000,000-token context and
at most 4,096 images. Native context ratios remain 0.8/0.1. Input images are not resized; MPO inputs retain
the existing primary-picture PNG conversion. Published path aliases are resolved by unique image basename.

The official QA template wraps the question with evidence-tool instructions. Only the isolated
[MobileMem Judge](../../plugins/mobilemem-judge/README.md) receives references and evidence. Its default is the
MemBase executable-default `mobilemem-omni-judge` rubric, `qwen3.8-max`, temperature 0, thinking disabled and
4,096 output tokens. Scores report binary LLM-Judge accuracy and the official token-overlap F1 and BLEU-1.

## Results

Each user's `workspace/` retains notes, source JSONL, images and native AgentScope trajectories.
`sessions/*.json` and `questions/*/` retain responses, judgments, metrics and technical errors.
`summary.json` reports completed/expected questions and scores over successfully scored results only.
Native interrupted replies and failed Jobs remain unscored technical gaps, not zero-score answers.
The runner adds no fallback or retry queue.

Usage policy version 2 counts observed successful HTTP requests, including compression and requests made before
a candidate history or answer later fails. Memory cost comes from `sessions/*.json` and `index-response.json`;
answer cost comes only from `answer-response.json`, including failed responses or `response: null` when a call raises.
Successful and failed answers use this single response file; its `response.success` determines whether scoring proceeds.
Earlier runs' `answer.json` copies, result files and response metadata are not counted again. `memory_usage.json` remains a
completion marker: it is written only after the full history and index update succeed.
Judge usage is excluded; absent counters stay unknown and each counter includes its observation coverage.
The `mean_tokens_per_user` / `mean_tokens_per_question` fields amortize this run's observed cost over scheduled users /
expected questions, with explicit denominators and attempt/completion counts. They are not successful-sample averages.
Earlier selected-success-only reports retain their original accounting policy.

Errors retain the failing stage, user/session/question identifiers, safe image paths, sanitized reasons and stack
locations; credentials, request headers, image payloads and frame locals are omitted. Cleanup errors are separate;
`cleanup_error_count` makes the run exit with status 2 without discarding already scored results.
`judge-settings.json` records a credential-free whitelist of the validated Judge model, parameters and protocol,
including an optional `JUDGE_MODEL_NAME` override, so scoring can be traced to the effective configuration.

Sources: [MobileMem evaluation code](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval)
and [MemBase judge](https://github.com/zjunlp/MemBase/blob/a9ba9495d48541d3e68588ff097294cc7a1201bc/membase/inference_utils/prompts.py).
Dataset revision: `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`.

## Upstream code license

The QA template and overlap metrics in `run.py` are adapted from ZJUNLP MobileMem's evaluation code at
`fa5cd13c9c659699df9caf650987ba2bea8ca085`, under the following license.

MIT License

Copyright (c) 2026 ZJUNLP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
