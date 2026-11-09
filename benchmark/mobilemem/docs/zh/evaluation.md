# MobileMem-Omni 视觉推理评测说明

本文介绍[多模态记忆文章](../../../../docs/zh/blog_20261009.md#mobilemem-results)中 MobileMem 的实验设置和结果。ReMe Image 的运行方式见 [benchmark README](../../README_ZH.md)。

## 结果与统计范围 {#results}

评测使用 MobileMem-Omni 官方 `visual_reasoning` 子集：**20 个用户、4,057 个 session、194,792 条消息和 1,333 个问题**。其他题型不进入本次比较。每个用户使用独立记忆工作空间；处理历史后，每题按官方 QA 模板独立回答。

数据版本为 `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`，QA 模板和指标实现对应[官方评测代码](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval)。

表中各项指标先逐题计算，再对全部 **1,333 题**的得分取平均（macro average）。LLM-Judge 为二元准确率，F1、BLEU-1 使用官方 token 重叠实现。Qwen ReMe Image 有 1 题、Text 有 3 题未完成：它们已完成历史处理，但在回答阶段多轮检索、读取后，一次后续模型请求的累积输入超过服务允许的 983,616 tokens 上限而被拒绝。各题检索到的材料和轮数不同，因此只有这几题触发限制。这四题在汇总中按 0 计，表中单列有效结果数。

| 模型 | 方法 | 有效结果数 / 1,333 | Judge 正确数 / 1,333 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | Full Context Image | 1,333 | 624 / 1,333 | 46.81 | 8.35 | 4.58 |
| Qwen3.8-Max | ReMe Text | 1,330 | 511 / 1,333 | 38.33 | 4.53 | 2.39 |
| Qwen3.8-Max | ReMe Image | 1,332 | 809 / 1,333 | **60.69** | 6.52 | 3.48 |
| Kimi-K3 | ReMe Text | 1,333 | 478 / 1,333 | 35.86 | 5.52 | 2.97 |
| Kimi-K3 | ReMe Image（由于内容审查跳过会话的历史变体） | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| DeepSeek-V4.1-Flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| DeepSeek-V4.1-Flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

表中 Kimi-K3 的“由于内容审查跳过会话的历史变体”指：最后 **130 题**对应的两位用户历史，分别排除了 `uuid_8` 的 `8_69`（原始顺序从 0 计数的位置 127）和 `uuid_14` 的 `14_74_2`（位置 116）。这两个 session 因内容审查未能完成记忆摄入。其余 **1,203 题**的已完成结果保留，全部 1,333 题仍纳入评测。其他 session 保持原顺序，图文输入、提示词、模型生成与评分参数，以及 30 轮回答上限一致。每题采用首次成功完成的回答和评分，0 分结果也纳入统计。

ReMe Image 在三个模型上的 LLM-Judge 准确率均高于 ReMe Text。在 Qwen3.8-Max 上，Image 还比 Full Context 高 13.88 个百分点；Full Context 的 F1、BLEU-1 则更高。

Kimi、DeepSeek 的 Full Context 均为 **N/A**：完整历史超过所用服务每请求最多 256 张图片的限制，因此未继续评测。

## 方法与 Judge 设置 {#evaluation-settings}

| 方法 | 记忆输入与回答阶段的数据访问范围 |
| --- | --- |
| Full Context Image | 直接读取按时间排序的完整对话和历史图片。图片缩放为 256 × 256 像素的 RGB PNG，使用 LANCZOS 重采样。 |
| ReMe Text | 使用原始对话文字构建记忆；通过 BM25 检索 daily、digest 笔记和原始 session。 |
| ReMe Image | 使用原始对话文字和图片构建记忆；检索范围与 Text 相同，并可以按需通过 `read_image` 重读附件。输入保留原分辨率；MPO 主图转换为 PNG。 |

除上述 Kimi Image 的两段会话排除外，历史文字、时间戳、角色和图片顺序均保留。数据集另附的 caption、摘要、参考答案和证据标注不进入候选模型输入。回答保留官方 QA 提示词。

| 设置 | ReMe 记忆／压缩 | ReMe 回答 | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| 模型 | 表中候选模型 | 同一候选模型 | 表中候选模型 | Qwen3.8-Max |
| Thinking | 开启 | 开启 | 开启 | 关闭 |
| Temperature | 0 | 0 | 0 | 0 |
| 输出预算（tokens） | 65,536 | 65,536 | 65,536 | 4,096 |
| 上下文上限（tokens） | 1,000,000 | 1,000,000 | — | — |

ReMe 每题新建回答 Agent，最多 **30 轮**，可以通过 `search`、`read` 检索和读取文本，并在需要时压缩上下文。Image 还可以通过 `read_image` 回看图片，读取上限为 50 MiB，Agent 上下文最多保留 4,096 张图。针对部分可能存在的偶发失败的 API 调用，实验中对其重试调用以获得有效结果。

全部结果统一使用 **MemBase 默认 Judge** 提示词及解析规则，模型和参数见上表。协议名为 `membase_default`，实现见 [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md)。参考答案和证据标注只提供给 Judge。

## 独立复现结果 {#independent-reproduction}

我们使用 main 分支（提交 [b784468](https://github.com/agentscope-ai/ReMe/commit/b784468a10d98af1688b7c4ba38492dc1f9bda6f)）中的[评测入口](https://github.com/agentscope-ai/ReMe/blob/b784468a10d98af1688b7c4ba38492dc1f9bda6f/benchmark/mobilemem/README.md)复现 ReMe Image。原图处理、BM25 检索、记忆与回答预算、30 轮上限和 MemBase Judge 设置沿用上文。

由于 [PR #713](https://github.com/agentscope-ai/ReMe/pull/713) 更新了 Agent 返回值的处理方式，达到轮次上限但未发生实际错误的结果也会被标记为失败，导致记忆摄入中止或回答无法进行评分。因此，我们在 [PR #735](https://github.com/agentscope-ai/ReMe/pull/735)（修复提交 [681a1383](https://github.com/agentscope-ai/ReMe/commit/681a13831426c1a8d4f7227ef835c38d0257003a)）中，仅针对 benchmark 插件重载结果处理，接纳达到 30 轮上限、无原生错误且无结构化输出约束的旧返回值（`exceed_max_iters`）；轮数上限和模型调用保持不变。

下表仅对**已评分题**求平均，未评分题不补 0，应测题数仍为 **1,333**。

| 模型 | 已评分 / 应测 | Judge 正确数 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | 1,329 / 1,333 | 803 | 60.42 | 6.52 | 3.49 |
| Kimi-K3 | 1,278 / 1,333 | 669 | 52.35 | 7.45 | 4.03 |
| DeepSeek-V4.1-Flash | 1,328 / 1,333 | 712 | 53.61 | 10.34 | 6.05 |

Kimi 的 1,278 题由 **1,148 题完整历史**（Judge 51.22%）和 **130 题由于内容审查跳过会话的历史变体**（62.31%）组成。后者为 `uuid_8` 的 67 题和 `uuid_14` 的 63 题，仍分别省略因内容审查摄入失败的 `8_69`、`14_74_2`。另一位用户 `uuid_0` 的历史在记忆生成阶段触发输出内容审核，其 55 题未进行问答评分。

Qwen 的 4 道未评分题在回答阶段因累积输入超过服务长度上限而被拒绝；DeepSeek 的 5 题则受同一历史会话 `8_69` 的输入图像内容审核失败影响。

与正式实验比较时，仅取双方均完成且问题 ID、问题及参考答案一致的共同题；差值由未舍入分数计算，单位为百分点。

| 模型 | 共同题数 | 正式实验 LLM-Judge (%) | 独立复现 LLM-Judge (%) | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | 1,328 | 60.92 | 60.47 | −0.45 |
| Kimi-K3 | 1,278 | 51.10 | 52.35 | +1.25 |
| DeepSeek-V4.1-Flash | 1,328 | 56.33 | 53.61 | −2.71 |

Kimi 的共同题也按历史输入分开比较：

| 历史输入 | 共同题数 | 正式实验 LLM-Judge (%) | 独立复现 LLM-Judge (%) | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 完整历史 | 1,148 | 50.61 | 51.22 | +0.61 |
| 由于内容审查跳过上述两个 session 的历史变体 | 130 | 55.38 | 62.31 | +6.92 |

就 ReMe Image 的 LLM-Judge 表现而言，复现结果与正式实验结果的趋势和结论保持一致。
