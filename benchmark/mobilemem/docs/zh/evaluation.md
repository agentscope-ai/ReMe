# MobileMem-Omni 视觉推理评测说明

本文介绍[多模态记忆文章](../../../../docs/zh/blog_20261009.md#mobilemem-results)中 MobileMem 的实验设置和结果。ReMe Image 的运行方式见 [benchmark README](../../README_ZH.md)。

## 结果与统计范围 {#results}

评测使用 MobileMem-Omni 官方 `visual_reasoning` 子集：**20 个用户、4,057 个 session、194,792 条消息和 1,333 个问题**。其他题型不进入本次比较。每个用户使用独立记忆工作空间；处理历史后，每题按官方 QA 模板独立回答。

数据版本为 `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`，QA 模板和指标实现对应[官方评测代码](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval)。

所有百分比均以相同的 **1,333 题**求宏平均。LLM-Judge 为二元准确率，F1、BLEU-1 使用官方 token 重叠实现。Qwen ReMe Image 有 1 题、Text 有 3 题未完成：它们已完成历史处理，但在回答阶段多轮检索、读取后，一次后续模型请求的累积输入超过服务允许的 983,616 tokens 上限而被拒绝。各题检索到的材料和轮数不同，因此只有这几题触发限制。这四题在汇总中按 0 计，表中单列有效结果数。

| 模型 | 方法 | 有效结果数 / 1,333 | Judge 正确数 / 1,333 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | Full Context Image | 1,333 | 624 / 1,333 | 46.81 | 8.35 | 4.58 |
| Qwen3.8-Max | ReMe Text | 1,330 | 511 / 1,333 | 38.33 | 4.53 | 2.39 |
| Qwen3.8-Max | ReMe Image | 1,332 | 809 / 1,333 | **60.69** | 6.52 | 3.48 |
| Kimi-K3 | ReMe Text | 1,333 | 478 / 1,333 | 35.86 | 5.52 | 2.97 |
| Kimi-K3 | ReMe Image（历史变体） | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| DeepSeek-V4.1-Flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| DeepSeek-V4.1-Flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

ReMe Image 在三个模型上的 LLM-Judge 准确率均高于 ReMe Text。在 Qwen3.8-Max 上，Image 还比 Full Context 高 13.88 个百分点；Full Context 的 F1、BLEU-1 则更高。

Kimi、DeepSeek 的 Full Context 均为 **N/A**：完整历史超过所用服务每请求最多 256 张图片的限制，因此未继续评测。

## 方法与 Judge 设置 {#evaluation-settings}

| 方法 | 记忆输入与回答阶段的数据访问范围 |
| --- | --- |
| Full Context Image | 直接读取按时间排序的完整对话和历史图片。图片缩放为 256 × 256 像素的 RGB PNG，使用 LANCZOS 重采样。 |
| ReMe Text | 使用原始对话文字构建记忆；通过 BM25 检索 daily、digest 笔记和原始 session。 |
| ReMe Image | 使用原始对话文字和图片构建记忆；检索范围与 Text 相同，并可以按需通过 `read_image` 重读附件。输入保留原分辨率；MPO 主图转换为 PNG。 |

历史文字、时间戳、角色和图片顺序均保留，Kimi Image 的两段会话排除另见下文。数据集另附的 caption、摘要、参考答案和证据标注不进入候选模型输入。回答保留官方 QA 提示词。

| 设置 | ReMe 记忆／压缩 | ReMe 回答 | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| 模型 | 表中候选模型 | 同一候选模型 | 表中候选模型 | Qwen3.8-Max |
| Thinking | 开启 | 开启 | 开启 | 关闭 |
| Temperature | 0 | 0 | 0 | 0 |
| 输出预算（tokens） | 65,536 | 65,536 | 65,536 | 4,096 |
| 上下文上限（tokens） | 1,000,000 | 1,000,000 | — | — |

ReMe 每题新建回答 Agent，最多 **30 轮**，可以通过 `search`、`read` 检索和读取文本，并在需要时压缩上下文。Image 还可以通过 `read_image` 回看图片，读取上限为 50 MiB，Agent 上下文最多保留 4,096 张图。

全部结果统一使用 **MemBase 默认 Judge** 提示词及解析规则，模型和参数见上表。协议名为 `membase_default`，实现见 [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md)。参考答案和证据标注只提供给 Judge。

## Kimi Image 历史变体 {#kimi-history-variant}

Kimi Image 最终结果保留**此前成功的 1,203 份答案**，补齐**两个用户的其余 130 题**。这两个用户的记忆摄入分别排除 `uuid_8` 的 `8_69`（原始顺序从 0 计数的位置 127）和 `uuid_14` 的 `14_74_2`（位置 116）。全部 1,333 题仍纳入评测，其他 session 保持原顺序；表中将该行标为历史变体。

除上述两段会话排除外，图文输入方式、提示词、模型生成与评分参数，以及 30 轮回答上限保持一致。每题采用首次成功完成的回答和评分，0 分结果也纳入统计。
