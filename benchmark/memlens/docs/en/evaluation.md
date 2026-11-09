# MemLens-256K Evaluation Notes

This report describes the settings, data selection and four-model results behind the [multimodal memory article](../../../../docs/en/blog_20261009.md#memlens-256k-results). See the [benchmark README](../../README.md) for running ReMe Image.

## Results and Evaluation Set {#results}

The evaluation uses the 195 questions in the official [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS) `agent_subset_195.json`, each with its 256K history. Six questions triggered provider content moderation in at least one method's model requests. All four models and all methods exclude these six questions so that every row reports results on the same **189/195 questions**.

The official category/subtype-specific LLM-Judge assigns each valid answer 0 or 1; accuracy is `100 × correct / 189`. SubEM measures whether the normalized reference is a substring of the prediction, and F1 measures normalized token overlap. Each metric is calculated per question, then averaged over all 189 questions (macro average). Missing Kimi and DeepSeek results count as zero in the aggregate, with valid coverage listed separately. Existing answers and scores, including zero scores, keep their original values.

| Model | Method | Valid / 189 | SubEM (%) | F1 (%) | Judge correct / 189 | LLM-Judge (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | Full Context Image | 189 | 39.68 | 42.08 | 89 / 189 | 47.09 |
| Qwen3.6-35B-A3B | ReMe Text | 189 | 44.44 | 38.58 | 78 / 189 | 41.27 |
| Qwen3.6-35B-A3B | ReMe Image | 189 | 53.97 | 47.54 | 101 / 189 | **53.44** |
| Qwen3.6-35B-A3B | OpenViking | 189 | 40.21 | 36.07 | 70 / 189 | 37.04 |
| Qwen3.6-35B-A3B | Mem0 | 189 | 30.69 | 26.54 | 73 / 189 | 38.62 |
| Qwen3.8-Max | Full Context Image | 189 | 60.32 | 57.74 | 124 / 189 | 65.61 |
| Qwen3.8-Max | ReMe Text | 189 | 50.26 | 44.61 | 100 / 189 | 52.91 |
| Qwen3.8-Max | ReMe Image | 189 | 64.55 | 57.85 | 137 / 189 | **72.49** |
| Qwen3.8-Max | OpenViking | 189 | 43.92 | 39.60 | 80 / 189 | 42.33 |
| Qwen3.8-Max | Mem0 | 189 | 38.10 | 39.79 | 97 / 189 | 51.32 |
| Kimi-K3 | Full Context Image | 184 | 53.97 | 52.07 | 108 / 189 | 57.14 |
| Kimi-K3 | ReMe Text | 189 | 50.26 | 40.81 | 90 / 189 | 47.62 |
| Kimi-K3 | ReMe Image | 184 | 62.43 | 53.51 | 119 / 189 | **62.96** |
| Kimi-K3 | OpenViking | 186 | 51.32 | 46.52 | 91 / 189 | 48.15 |
| Kimi-K3 | Mem0 | 184 | 40.74 | 34.94 | 98 / 189 | 51.85 |
| DeepSeek-V4.1-Flash | Full Context Image | 187 | 57.67 | 55.70 | 114 / 189 | 60.32 |
| DeepSeek-V4.1-Flash | ReMe Text | 185 | 42.86 | 37.78 | 78 / 189 | 41.27 |
| DeepSeek-V4.1-Flash | ReMe Image | 186 | 58.20 | 53.59 | 117 / 189 | **61.90** |
| DeepSeek-V4.1-Flash | OpenViking | 185 | 50.79 | 45.12 | 93 / 189 | 49.21 |
| DeepSeek-V4.1-Flash | Mem0 | 183 | 41.80 | 35.81 | 93 / 189 | 49.21 |

ReMe Image has the highest LLM-Judge accuracy among the five methods for each candidate model.

## Method and Judge Settings {#evaluation-settings}

| Method | Memory construction | Answering |
| --- | --- | --- |
| Full Context Image | None. | The model reads the complete conversation text and original images, then answers directly. |
| ReMe Text | Uses ReMe's `auto_memory` with the original conversation text. | A fresh `agentic_answer` Agent searches daily notes, digest notes and original sessions with BM25, then reads relevant text over multiple rounds. |
| ReMe Image | Uses the same memory workflow with original text and images. | Uses the same Agent workflow as Text, with `read_image` available to reopen saved original images. |
| OpenViking | Uses its native image descriptions and session-memory workflow. | One native `search` call, restricted to memory with the SDK's default top 10, followed by one independent reader completion. |
| Mem0 | Uses Mem0 2.0.4's native vision and memory extraction. | One native `search` call with FAISS top 20, followed by one independent reader completion. |

The baselines retain their own memory construction and retrieval workflows. OpenViking and Mem0 use retrieve → answer; ReMe uses its multi-round `agentic_answer` workflow. Each row uses the named candidate model for answering, and memory construction uses the same model where applicable. ReMe Image and Mem0 receive original images rather than supplied benchmark captions; OpenViking produces its own image descriptions.

On the evaluation server, the **OpenViking 0.4.17.2.dev1** main package was installed in editable mode from source, based on [this upstream revision](https://github.com/volcengine/OpenViking/tree/420d4074f74070bc6fe7151aa54527cf1ff15152), with Python SDK 0.1.9. It formats retrieved text with MemLens's flat QA template. **Mem0 2.0.4** was also installed in editable mode from its [pinned source revision](https://github.com/mem0ai/mem0/tree/75a37ec93db7278e3bd9aaf2aa3d6e5139e6789d), following the [MemLens reproduction instructions](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/README.md#mem0) and its dedicated [QA message builder](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/prompt_builders.py).

| Stage | Context limit | Output budget | Thinking / temperature |
| --- | ---: | ---: | --- |
| ReMe memory / compression | 200,000 | 65,536 | On / 0 |
| ReMe answer | 400,000 | 16,384 | On / 0 |
| Full Context, OpenViking and Mem0 answer | — | 16,384 | On / 0 |
| MemLens Judge | — | 256 | Off / 0 |

Each ReMe question uses an independent answer Agent with at most 30 iterations and context compression. Its `read_image` adapter accesses saved attachments with a 50 MiB read limit and a 4,096-image retention limit. Some API calls were retried after potentially transient failures to obtain valid results.

Mem0's memory-extraction output budget is **2,000 tokens for Qwen and 10,000 for Kimi/DeepSeek**. Qwen and DeepSeek inherit Mem0 2.0.4's [OpenAI adapter defaults](https://github.com/mem0ai/mem0/blob/75a37ec93db7278e3bd9aaf2aa3d6e5139e6789d/mem0/configs/llms/openai.py): `temperature=0.1` and `top_p=0.1`. These values come from Mem0's configuration. The service used for Kimi does not accept these parameters, so they are omitted from its requests; the provider's actual defaults are unknown. Memory-extraction requests do not explicitly set a thinking switch and use the model service's default behavior; answering explicitly enables thinking as shown above.

Scoring follows the [official MemLens protocol](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2). All methods use **Qwen3.8-Max** as Judge, with temperature **0**, thinking **disabled** and a **256-token** output budget. The Judge runs independently of candidate memory applications; reference answers and question-type labels are used only for scoring.

## Common Content-Moderation Filter {#content-moderation-filter}

The table lists six questions that triggered content moderation with Qwen3.6-35B-A3B or Qwen3.8-Max during memory construction or answering, along with the affected methods. All four models and all methods exclude these six questions and report results on the remaining 189.

In the table, A3B denotes Qwen3.6-35B-A3B and Max denotes Qwen3.8-Max.

| Question ID | Method / model that triggered moderation | Stage |
| --- | --- | --- |
| `q_028cb832` | OpenViking / Max | Memory construction |
| `q_1d99a36c` | OpenViking / A3B | Memory construction |
| `q_3b01c02a` | ReMe Text and Image / A3B; OpenViking / A3B and Max; Mem0 / A3B | Memory construction |
| `q_564a657b` | ReMe Image / Max | Answering |
| `q_600a79aa` | ReMe Image / Max | Memory construction |
| `q_c9f8e24d` | ReMe Text / Max | Answering |

## Independent Reproduction Results {#independent-reproduction}

The reproduction is based on the main branch at [b784468](https://github.com/agentscope-ai/ReMe/commit/b784468a10d98af1688b7c4ba38492dc1f9bda6f) and uses the repository's [benchmark entry point](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/benchmark/memlens/README.md). The [method configuration](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/plugins/memlens/src/reme_memlens/benchmark.yaml), [Judge configuration](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/plugins/memlens-judge/src/judge_memlens/plugin.yaml) and [Judge scoring rules](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/plugins/memlens-judge/src/judge_memlens/protocol.py) specify context and output budgets, the 30-iteration limit and scoring protocol. Original-image handling and BM25 retrieval follow that revision's benchmark implementation.

[PR #713](https://github.com/agentscope-ai/ReMe/pull/713) introduced `AgentReply` handling that classifies native `exceed_max_iters` returns as failures even when the native `error` is `None`. This can stop memory ingestion or keep an answer from being scored. The fix proposed in [PR #735](https://github.com/agentscope-ai/ReMe/pull/735) ([681a1383](https://github.com/agentscope-ai/ReMe/commit/681a13831426c1a8d4f7227ef835c38d0257003a)) overrides result handling only in the benchmark plugins: it accepts exactly `exceed_max_iters` with native `error=None` and no requested `output_schema`, preserving the original text, usage and diagnostics without adding iterations or model calls.

All 195 questions are supplied as input; the main set excludes the same six questions listed above. Scores below average **scored questions only**, with no zero fill for unscored questions. Five DeepSeek questions remained unscored because input images triggered content moderation during memory ingestion.

| Model | Scored / expected | Judge correct | LLM-Judge (%) | SubEM (%) | F1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 189 / 189 | 104 | 55.03 | 54.50 | 48.01 |
| Qwen3.8-Max | 189 / 189 | 133 | 70.37 | 61.38 | 55.52 |
| Kimi-K3 | 189 / 189 | 123 | 65.08 | 65.61 | 55.81 |
| DeepSeek-V4.1-Flash | 184 / 189 | 119 | 64.67 | 59.24 | 56.23 |

Comparisons with the main experiment use questions completed in both runs with matching IDs, questions and references. Both score columns below are recalculated on each row's common set. Differences use unrounded values and are in percentage points.

| Model | Common questions | Main experiment LLM-Judge (%) | Reproduction LLM-Judge (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 189 | 53.44 | 55.03 | +1.59 |
| Qwen3.8-Max | 189 | 72.49 | 70.37 | −2.12 |
| Kimi-K3 | 184 | 64.67 | 65.22 | +0.54 |
| DeepSeek-V4.1-Flash | 181 | 64.09 | 64.64 | +0.55 |

ReMe Image's model rankings by LLM-Judge are unchanged. On this measure, the reproduction results are consistent with the trends and conclusions of the main experiment.
