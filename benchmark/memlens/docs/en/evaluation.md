# MemLens-256K Evaluation Notes

This report describes the settings, data selection and four-model results behind the [multimodal memory article](../../../../docs/en/blog_20261009.md#memlens-256k-results). See the [benchmark README](../../README.md) for running ReMe Image.

## Results and Evaluation Set {#results}

The evaluation uses the 195 questions in the official [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS) `agent_subset_195.json`, each with its 256K history. The six-question moderation filter established in the original Qwen experiment is fixed for all four models, leaving the same **189/195 questions** for every row.

The official category/subtype-specific LLM-Judge assigns each valid answer 0 or 1; accuracy is `100 × correct / 189`. SubEM measures whether the normalized reference is a substring of the prediction, and F1 measures normalized token overlap. All metrics are question-level macro averages. Missing Kimi and DeepSeek results count as zero in the 189-question aggregate, with valid coverage listed separately. Existing answers and scores, including zero scores, keep their original values.

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
| kimi-k3 | Full Context Image | 184 | 53.97 | 52.07 | 108 / 189 | 57.14 |
| kimi-k3 | ReMe Text | 189 | 50.26 | 40.81 | 90 / 189 | 47.62 |
| kimi-k3 | ReMe Image | 184 | 62.43 | 53.51 | 119 / 189 | **62.96** |
| kimi-k3 | OpenViking | 186 | 51.32 | 46.52 | 91 / 189 | 48.15 |
| kimi-k3 | Mem0 | 184 | 40.74 | 34.94 | 98 / 189 | 51.85 |
| deepseek-v4.1-flash | Full Context Image | 187 | 57.67 | 55.70 | 114 / 189 | 60.32 |
| deepseek-v4.1-flash | ReMe Text | 185 | 42.86 | 37.78 | 78 / 189 | 41.27 |
| deepseek-v4.1-flash | ReMe Image | 186 | 58.20 | 53.59 | 117 / 189 | **61.90** |
| deepseek-v4.1-flash | OpenViking | 185 | 50.79 | 45.12 | 93 / 189 | 49.21 |
| deepseek-v4.1-flash | Mem0 | 183 | 41.80 | 35.81 | 93 / 189 | 49.21 |

ReMe Image has the highest LLM-Judge accuracy among the five methods for each candidate model. The original Qwen per-model charts remain available: [Qwen3.6-35B-A3B](../figure/memlens-256k-qwen36-judge.svg) and [Qwen3.8-Max](../figure/memlens-256k-qwen38-judge.svg).

## Method and Judge Settings {#evaluation-settings}

| Method | Memory construction | Answering |
| --- | --- | --- |
| Full Context Image | None. | The model reads the complete conversation text and original images, then answers directly. |
| ReMe Text | Uses ReMe's `auto_memory` with the original conversation text. | A fresh `agentic_answer` Agent searches daily notes, digest notes and original sessions with BM25, then reads relevant text over multiple rounds. |
| ReMe Image | Uses the same memory workflow with original text and images. | Uses the same Agent workflow as Text, with `read_image` available to reopen saved original images. |
| OpenViking | Uses its native image descriptions and session-memory workflow. | One native `search` call, restricted to memory with the SDK's default top 10, followed by one independent reader completion. |
| Mem0 | Uses Mem0 2.0.4's native vision and memory extraction. | One native `search` call with FAISS top 20, followed by one independent reader completion. |

The baselines retain their own memory construction and retrieval workflows. OpenViking and Mem0 use retrieve → answer; ReMe uses its multi-round `agentic_answer` workflow. Each row uses the named candidate model for answering, and memory construction uses the same model where applicable. ReMe Image and Mem0 receive original images rather than supplied benchmark captions; OpenViking produces its own image descriptions.

The OpenViking reproduction uses SDK 0.1.9, based on [this upstream revision](https://github.com/volcengine/OpenViking/tree/420d4074f74070bc6fe7151aa54527cf1ff15152), and formats retrieved text with MemLens's flat QA template. Mem0 follows the [MemLens reproduction instructions](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/README.md#mem0), using [Mem0 2.0.4](https://github.com/mem0ai/mem0/tree/75a37ec93db7278e3bd9aaf2aa3d6e5139e6789d) and its dedicated [QA message builder](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/prompt_builders.py). The retrieval scope, model settings and adapter changes used in this evaluation are recorded here alongside those upstream workflows.

| Stage | Context limit | Output budget | Thinking / temperature |
| --- | ---: | ---: | --- |
| ReMe memory / compression | 200,000 | 65,536 | On / 0 |
| ReMe answer | 400,000 | 16,384 | On / 0 |
| Full Context, OpenViking and Mem0 answer | — | 16,384 | On / 0 |
| MemLens Judge | — | 256 | Off / 0 |

Each ReMe question uses an independent answer Agent with at most 30 iterations and native context compression. Its `read_image` adapter accesses saved attachments with a 50 MiB read limit and a 4,096-image retention limit. The OpenViking adapter includes a JSON parsing correction and explicit image-processing errors.

Mem0's native vision/extraction output budget is **2,000 tokens for Qwen and 10,000 for Kimi/DeepSeek**. Qwen and DeepSeek use native temperature/top_p 0.1/0.1. Kimi omits these native sampling parameters because the service does not support them; the provider defaults are unknown. Native thinking is not overridden.

Scoring follows the [official MemLens protocol](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2). All methods use **Qwen3.8-Max** as Judge, with temperature **0**, thinking **disabled** and a **256-token** output budget. The Judge runs independently of candidate memory applications; reference answers and question-type labels are used only for scoring.

## Common Content-Moderation Filter {#content-moderation-filter}

The method/question combinations below triggered content moderation in the original Qwen experiment. Taking the union across its methods and two models gives six questions. This same set is excluded from every method and all four models; the Kimi/DeepSeek results do not expand the filter.

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

We organized and integrated the MemLens benchmark code into ReMe, then independently reran ReMe Image using the same data inputs and scoring settings. See the [run instructions](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/benchmark/memlens/README.md), [method configuration](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens/src/reme_memlens/benchmark.yaml), [Judge configuration](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens-judge/src/judge_memlens/plugin.yaml) and [Judge scoring rules](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens-judge/src/judge_memlens/protocol.py) for the implementation and settings.

The corresponding public evaluation code is available in [PR #640](https://github.com/agentscope-ai/ReMe/pull/640) at commit [962d5cd](https://github.com/agentscope-ai/ReMe/commit/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8).

After applying the same six-question filter, one A3B question and two Max questions remain incomplete because conversation-history processing failed. The table compares the 188 and 187 questions completed in both experiments, respectively.

| Model | Paired questions | Original ReMe Image correct | Independent reproduction correct | Original → reproduction LLM-Judge (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 188 | 100 | 96 | 53.19 → 51.06 |
| Qwen3.8-Max | 187 | 135 | 134 | 72.19 → 71.66 |

On these matched questions, the reproduced LLM-Judge scores for Qwen3.6-35B-A3B and Qwen3.8-Max are 2.13 and 0.53 percentage points lower than the original results, respectively. Overall, performance remains broadly consistent with the original experiment.

## Result Sources

The main table combines the archived Qwen experiment (`final-20261017`) with the Kimi/DeepSeek extension (`append-20261023-kimi-flash`). The latter's `statistics/memlens-fixed189.json` records all 20 rows, fixed question IDs, coverage and existing per-question metrics. Results were aggregated without generating new answers or rerunning the Judge. The independent reproduction remains a separate paired comparison above.
