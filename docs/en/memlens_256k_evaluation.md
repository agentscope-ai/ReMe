# MemLens-256K Evaluation Notes

This report describes the experimental settings, data selection and results of the MemLens-256K evaluation in the [multimodal memory article](./blog_20261009.md#memlens-256k-results).

## Results and Evaluation Set {#results}

The evaluation uses the 195 questions in the official [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS) `agent_subset_195.json`, each with its complete 256K history. After excluding the six questions listed below, all methods are evaluated on the same 189 questions. The official category/subtype-specific LLM-Judge protocol assigns each answer a score of 0 or 1; the reported percentage is `100 × correct / 189`.

| Model | Method | Judge correct / 189 | LLM-Judge (%) |
| --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B | Full Context Image | 89 / 189 | 47.09 |
| Qwen3.6-35B-A3B | ReMe Text | 78 / 189 | 41.27 |
| Qwen3.6-35B-A3B | ReMe Image | 101 / 189 | 53.44 |
| Qwen3.6-35B-A3B | OpenViking | 70 / 189 | 37.04 |
| Qwen3.6-35B-A3B | Mem0 | 73 / 189 | 38.62 |
| Qwen3.8-Max | Full Context Image | 124 / 189 | 65.61 |
| Qwen3.8-Max | ReMe Text | 100 / 189 | 52.91 |
| Qwen3.8-Max | ReMe Image | 137 / 189 | 72.49 |
| Qwen3.8-Max | OpenViking | 80 / 189 | 42.33 |
| Qwen3.8-Max | Mem0 | 97 / 189 | 51.32 |

ReMe Image scores **53.44%** and **72.49%** on the two models, exceeding Full Context Image by **6.35** and **6.88** percentage points, respectively.

## Method and Judge Settings {#evaluation-settings}

| Method | Memory input and answer-time data access |
| --- | --- |
| Full Context Image | The answer model receives the complete conversation text and original images directly. |
| ReMe Text | Builds memory from the original conversation text. The answer Agent uses BM25 to search daily notes, digest notes and original sessions, then reads relevant text. |
| ReMe Image | Builds memory from the original conversation text and images. Retrieval covers the same sources as Text, with `read_image` available to reopen saved original images. |
| OpenViking | Uses native image descriptions and memory processing. Answer-time retrieval is restricted to memory with the default top 10. |
| Mem0 | Builds memory through Mem0 2.0.4's native vision interface. Answer-time retrieval uses FAISS to select the top 20 memories. |

Each method is evaluated with Qwen3.6-35B-A3B and Qwen3.8-Max as the candidate models. ReMe Image and Mem0 both use original images. ReMe's memory and answer context limits are 200,000 and 400,000 tokens, with temperature 0 and thinking enabled. Memory/compression and answer output budgets are 65,536 and 16,384 tokens. Each question uses an independent answer Agent with at most 30 iterations and native context compression. The `read_image` adapter accesses saved attachments with a 50 MiB read limit and a 4,096-image retention limit.

Full Context, OpenViking and Mem0 also use temperature 0, thinking enabled and a 16,384-token output budget for answering. Mem0's memory stage uses its native settings: a 2,000-token output budget, temperature 0.1 and top_p 0.1. The OpenViking evaluation adapter includes a JSON parsing correction and explicit error reporting for image-processing failures.

Scoring follows the [official MemLens protocol](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2). All methods use **Qwen3.8-Max** as Judge, with temperature **0**, thinking **disabled** and a **256-token** output budget. The Judge runs independently of candidate memory applications; reference answers and question-type labels are used only for scoring.

## Common Content-Moderation Filter {#content-moderation-filter}

The method/question combinations below triggered content moderation when calling the Qwen model service. Taking the union of affected questions across all methods and both models gives six questions. We exclude this same set from every method and model, leaving **189/195** questions for evaluation.

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
