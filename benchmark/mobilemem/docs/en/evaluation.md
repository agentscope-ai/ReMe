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
| Kimi-K3 | ReMe Image (two sessions skipped due to content moderation) | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| DeepSeek-V4.1-Flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| DeepSeek-V4.1-Flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

The Kimi-K3 ReMe Image row includes **130 questions** whose histories omit two sessions because content moderation blocked memory ingestion: `8_69` of `uuid_8` (original zero-based position 127) and `14_74_2` of `uuid_14` (position 116). The completed results for the other **1,203 questions** are retained, and all 1,333 questions remain in the evaluation. Other sessions keep their original order; image and text handling, prompts, generation and scoring parameters, and the 30-iteration answer limit are unchanged. Each question uses the first successfully completed answer and judgment, including zero scores.

ReMe Image exceeds ReMe Text in LLM-Judge accuracy for all three models. On Qwen3.8-Max, it also exceeds Full Context by 13.88 percentage points, while Full Context has higher F1 and BLEU-1.

Kimi and DeepSeek Full Context are **N/A**: their complete histories exceeded the service's limit of 256 images per request, so these runs were not continued.

## Method and Judge Settings {#evaluation-settings}

| Method | Memory input and answer-time data access |
| --- | --- |
| Full Context Image | Receives the complete chronological dialogue and historical images directly. Images are resized to 256 × 256-pixel RGB PNGs with LANCZOS resampling. |
| ReMe Text | Builds memory from the original dialogue text; BM25 searches daily notes, digest notes and original sessions. |
| ReMe Image | Builds memory from original dialogue text and images; searches the same sources as Text and can use `read_image` to reopen saved attachments as needed. Inputs retain original resolution; MPO primary pictures are converted to PNG. |

History text, timestamps, roles and image order are preserved apart from the two Kimi Image sessions skipped due to content moderation, as described above. Separate dataset captions, summaries, reference answers and evidence annotations are not added to candidate inputs. The official QA prompt is retained.

| Setting | ReMe memory / compression | ReMe answer | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| Model | Named candidate | Same candidate | Named candidate | Qwen3.8-Max |
| Thinking | On | On | On | Off |
| Temperature | 0 | 0 | 0 | 0 |
| Output budget (tokens) | 65,536 | 65,536 | 65,536 | 4,096 |
| Context limit (tokens) | 1,000,000 | 1,000,000 | — | — |

Each ReMe question starts a fresh Agent with at most **30 iterations**. The Agent can search and read text through `search` and `read`, and compress its context when needed. Image can also use `read_image` to reopen images, with a 50 MiB read limit and up to 4,096 images retained in the Agent context. Some API calls were retried after potentially transient failures to obtain valid results.

All rows use the default **MemBase Judge** prompt and parsing rules, with the Qwen3.8-Max scoring settings above. The protocol is named `membase_default`; see the [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md) for its implementation. Only the Judge receives reference answers and evidence annotations.

<a name="independent-reproduction"></a>

## Independent Reproduction Results {#independent-reproduction}

The reproduction is based on the main branch at [b784468](https://github.com/agentscope-ai/ReMe/commit/b784468a10d98af1688b7c4ba38492dc1f9bda6f) and uses the repository's [benchmark entry point](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/benchmark/mobilemem/README.md). Original-image handling, BM25 retrieval, memory and answer budgets, the 30-iteration limit and MemBase Judge settings follow the configuration above.

[PR #713](https://github.com/agentscope-ai/ReMe/pull/713) introduced `AgentReply` handling that classifies native `exceed_max_iters` returns as failures even when the native `error` is `None`. This can stop memory ingestion or keep an answer from being scored. The fix proposed in [PR #735](https://github.com/agentscope-ai/ReMe/pull/735) ([681a1383](https://github.com/agentscope-ai/ReMe/commit/681a13831426c1a8d4f7227ef835c38d0257003a)) overrides result handling only in the benchmark plugins: it accepts exactly `exceed_max_iters` with native `error=None` and no requested `output_schema`, without adding iterations or model calls.

Scores below average **scored questions only**, with no zero fill for unscored questions. The expected set remains **1,333 questions**.

| Model | Scored / expected | Judge correct | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | 1,329 / 1,333 | 803 | 60.42 | 6.52 | 3.49 |
| Kimi-K3 | 1,278 / 1,333 | 669 | 52.35 | 7.45 | 4.03 |
| DeepSeek-V4.1-Flash | 1,328 / 1,333 | 712 | 53.61 | 10.34 | 6.05 |

Kimi's 1,278 scored questions comprise **1,148 with complete histories** (Judge 51.22%) and **130 using histories with sessions skipped due to content moderation** (62.31%). The latter are 67 questions from `uuid_8` and 63 from `uuid_14`, omitting sessions `8_69` and `14_74_2`, respectively. Another user's history, `uuid_0`, triggered output moderation during memory generation, so that user's 55 questions did not proceed to answering or scoring.

Qwen's four unscored questions were rejected during answering because accumulated inputs exceeded the service's length limit. DeepSeek's five were affected by a single input-image moderation failure in history session `8_69`.

Comparisons with the main experiment use questions completed in both runs with matching IDs, questions and references. Differences use unrounded values and are in percentage points.

| Model | Common questions | Main experiment LLM-Judge (%) | Reproduction LLM-Judge (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | 1,328 | 60.92 | 60.47 | −0.45 |
| Kimi-K3 | 1,278 | 51.10 | 52.35 | +1.25 |
| DeepSeek-V4.1-Flash | 1,328 | 56.33 | 53.61 | −2.71 |

Kimi's common questions are also compared separately by history input:

| History input | Common questions | Main experiment LLM-Judge (%) | Reproduction LLM-Judge (%) | Difference (pp) |
| --- | ---: | ---: | ---: | ---: |
| Complete history | 1,148 | 50.61 | 51.22 | +0.61 |
| History with the two sessions above skipped due to content moderation | 130 | 55.38 | 62.31 | +6.92 |

ReMe Image's model rankings by LLM-Judge are unchanged. On this measure, the reproduction results are consistent with the trends and conclusions of the main experiment.
