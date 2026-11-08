# MobileMem-Omni Visual Reasoning Evaluation

This report describes the settings and results behind the [multimodal memory article](../../../../docs/en/blog_20261009.md#mobilemem-results). See the [benchmark README](../../README.md) for running ReMe Image.

## Results and Evaluation Set {#results}

The evaluation uses the official MobileMem-Omni `visual_reasoning` subset: **20 users, 4,057 sessions, 194,792 messages and 1,333 questions**. Other question types are outside this comparison. Each user has an isolated memory workspace; questions use the official QA template and are answered independently after history processing.

Dataset revision: `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`. The QA template and metrics follow the [official evaluation code](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval).

Each metric is calculated per question, then averaged over all **1,333 questions** (macro average). LLM-Judge is binary accuracy; F1 and BLEU-1 use the official token-overlap implementations. One Qwen ReMe Image question and three Text questions remained incomplete after successful history processing. After multiple rounds of retrieval and reading, a subsequent model request's accumulated input exceeded the service's 983,616-token limit and was rejected. Each question retrieves different material over a different number of rounds, so only these requests reached the limit. These four questions count as zero in the aggregate, with valid coverage listed separately.

| Model | Method | Valid / 1,333 | Judge correct / 1,333 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | Full Context Image | 1,333 | 624 / 1,333 | 46.81 | 8.35 | 4.58 |
| Qwen3.8-Max | ReMe Text | 1,330 | 511 / 1,333 | 38.33 | 4.53 | 2.39 |
| Qwen3.8-Max | ReMe Image | 1,332 | 809 / 1,333 | **60.69** | 6.52 | 3.48 |
| Kimi-K3 | ReMe Text | 1,333 | 478 / 1,333 | 35.86 | 5.52 | 2.97 |
| Kimi-K3 | ReMe Image (history variant) | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| DeepSeek-V4.1-Flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| DeepSeek-V4.1-Flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

The Kimi-K3 row is marked as a history variant because the histories used for the final **130 questions** omit two sessions whose memory ingestion was blocked by content moderation: `8_69` of `uuid_8` (original zero-based position 127) and `14_74_2` of `uuid_14` (position 116). The completed results for the other **1,203 questions** are retained, and all 1,333 questions remain in the evaluation. Other sessions keep their original order; image and text handling, prompts, generation and scoring parameters, and the 30-iteration answer limit are unchanged. Each question uses the first successfully completed answer and judgment, including zero scores.

ReMe Image exceeds ReMe Text in LLM-Judge accuracy for all three models. On Qwen3.8-Max, it also exceeds Full Context by 13.88 percentage points, while Full Context has higher F1 and BLEU-1.

Kimi and DeepSeek Full Context are **N/A**: their complete histories exceeded the service's limit of 256 images per request, so these runs were not continued.

## Method and Judge Settings {#evaluation-settings}

| Method | Memory input and answer-time data access |
| --- | --- |
| Full Context Image | Receives the complete chronological dialogue and historical images directly. Images are resized to 256 × 256-pixel RGB PNGs with LANCZOS resampling. |
| ReMe Text | Builds memory from the original dialogue text; BM25 searches daily notes, digest notes and original sessions. |
| ReMe Image | Builds memory from original dialogue text and images; searches the same sources as Text and can use `read_image` to reopen saved attachments as needed. Inputs retain original resolution; MPO primary pictures are converted to PNG. |

History text, timestamps, roles and image order are preserved apart from the two Kimi Image session exclusions described above. Separate dataset captions, summaries, reference answers and evidence annotations are not added to candidate inputs. The official QA prompt is retained.

| Setting | ReMe memory / compression | ReMe answer | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| Model | Named candidate | Same candidate | Named candidate | Qwen3.8-Max |
| Thinking | On | On | On | Off |
| Temperature | 0 | 0 | 0 | 0 |
| Output budget (tokens) | 65,536 | 65,536 | 65,536 | 4,096 |
| Context limit (tokens) | 1,000,000 | 1,000,000 | — | — |

Each ReMe question starts a fresh Agent with at most **30 iterations**. The Agent can search and read text through `search` and `read`, and compress its context when needed. Image can also use `read_image` to reopen images, with a 50 MiB read limit and up to 4,096 images retained in the Agent context.

All rows use the default **MemBase Judge** prompt and parsing rules, with the Qwen3.8-Max scoring settings above. The protocol is named `membase_default`; see the [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md) for its implementation. Only the Judge receives reference answers and evidence annotations.
