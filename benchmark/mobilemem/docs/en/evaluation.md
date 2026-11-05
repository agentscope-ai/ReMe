# MobileMem-Omni Visual Reasoning Evaluation

This report describes the settings and results behind the [multimodal memory article](../../../../docs/en/blog_20261009.md#mobilemem-results). See the [benchmark README](../../README.md) for running ReMe Image.

## Results and Evaluation Set {#results}

The evaluation uses the official MobileMem-Omni `visual_reasoning` subset: **20 users, 4,057 sessions, 194,792 messages and 1,333 questions**. Other question types are outside this comparison. Each user has an isolated memory workspace; questions use the official QA template and are answered independently after history processing.

All percentages are question-level macro averages over the same **1,333 questions**. LLM-Judge is binary accuracy; F1 and BLEU-1 use the official token-overlap implementations. Qwen ReMe Image has one input-length failure and Text has three; these questions count as zero in the aggregate, with valid coverage listed separately.

| Model | Method | Valid / 1,333 | Judge correct / 1,333 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | Full Context Image | 1,333 | 624 / 1,333 | 46.81 | 8.35 | 4.58 |
| Qwen3.8-Max | ReMe Text | 1,330 | 511 / 1,333 | 38.33 | 4.53 | 2.39 |
| Qwen3.8-Max | ReMe Image | 1,332 | 809 / 1,333 | **60.69** | 6.52 | 3.48 |
| kimi-k3 | ReMe Text | 1,333 | 478 / 1,333 | 35.86 | 5.52 | 2.97 |
| kimi-k3 | ReMe Image (history variant) | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| deepseek-v4.1-flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| deepseek-v4.1-flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

ReMe Image exceeds ReMe Text in LLM-Judge accuracy for all three models. On Qwen3.8-Max, it also exceeds Full Context by 13.88 percentage points, while Full Context has higher F1 and BLEU-1.

Kimi and DeepSeek Full Context are **N/A**: their complete histories exceeded the service's limit of 256 images per request, so these runs were not continued. Earlier OpenViking and Mem0 MobileMem runs are outside this comparison.

## Method and Judge Settings {#evaluation-settings}

| Method | Memory input and answer-time data access |
| --- | --- |
| Full Context Image | Receives the complete chronological dialogue and historical images directly. Images are resized to 256 × 256 RGB PNGs with LANCZOS resampling (`paper256`). |
| ReMe Text | Builds memory from the original dialogue text; BM25 searches daily notes, digest notes and original sessions. |
| ReMe Image | Builds memory from original dialogue text and images; searches the same sources as Text and uses `read_image` to reopen saved attachments. Inputs retain original resolution; MPO primary pictures are converted to PNG. |

History text, timestamps, roles and image order are preserved, subject to the declared Kimi Image exclusions below. Separate dataset captions, summaries, reference answers and evidence annotations are not added to candidate inputs. The official QA prompt is retained. ReMe Image's memory and answer Jobs receive images explicitly.

| Setting | ReMe memory / compression | ReMe answer | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| Model | Named candidate | Same candidate | Named candidate | Qwen3.8-Max |
| Thinking | On | On | On | Off |
| Temperature | 0 | 0 | 0 | 0 |
| Output budget (tokens) | 65,536 | 65,536 | 65,536 | 4,096 |
| Context limit (tokens) | 1,000,000 | 1,000,000 | — | — |

Each ReMe question starts a fresh Agent with at most **30 iterations**, native context compression and `search`, `read` and `read_image` tools. Original sessions remain searchable. The image adapter permits 50 MiB reads and retains up to 4,096 images in the Agent context.

All rows use the default **MemBase Judge** prompt and native parser, with the Qwen3.8-Max scoring settings above. The protocol is named `membase_default`; see the [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md) for its implementation. Qwen Full Context predictions were scored under this default protocol; earlier strict-rubric scores are outside this table. Only the Judge receives reference answers and evidence annotations.

## Kimi Image History Variant {#kimi-history-variant}

The final Kimi Image result retains **1,203 previously successful answers** and adds the remaining **130 questions from two users**. For those two users, memory ingestion excludes session `8_69` of `uuid_8` (original zero-based position 127) and `14_74_2` of `uuid_14` (position 116). All 1,333 questions remain in the evaluation, and other sessions keep their original order. This row is marked as a history variant in the table.

Kimi recovery used session checkpoints, identical-payload HTTP retries and increased question concurrency. Apart from the two declared session exclusions, input, prompts, reader/Judge budgets and the 30-iteration answer limit were retained. DeepSeek retried technical failures and content-moderation errors. Both models retained the first eligible result, including valid zero scores.

## Result Sources

The table combines the Qwen archive `final-20261017` and the Kimi/DeepSeek extension `append-20261023-kimi-flash`. The extension's `statistics/mobile-final.json` records final metrics and fixed denominators; `selected/mobile-selected.json` records per-question selection. Existing predictions and judgments were aggregated without new model calls. The dataset revision is `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`, and the [official evaluation code](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval) is fixed at `fa5cd13c9c659699df9caf650987ba2bea8ca085`.
