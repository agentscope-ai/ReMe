# MobileMem-Omni 视觉推理评测说明

本文介绍[多模态记忆文章](../../../../docs/zh/blog_20261009.md#mobilemem-results)中 MobileMem 的实验设置和结果。ReMe Image 的运行方式见 [benchmark README](../../README_ZH.md)。

## 结果与统计范围 {#results}

评测使用 MobileMem-Omni 官方 `visual_reasoning` 子集：**20 个用户、4,057 个 session、194,792 条消息和 1,333 个问题**。其他题型不进入本次比较。每个用户使用独立记忆工作空间；处理历史后，每题按官方 QA 模板独立回答。

所有百分比均以相同的 **1,333 题**求宏平均。LLM-Judge 为二元准确率，F1、BLEU-1 使用官方 token 重叠实现。Qwen ReMe Image 有 1 题、Text 有 3 题因输入超长未完成，在汇总中按 0 计，并单列有效结果数。

| 模型 | 方法 | 有效数 / 1,333 | Judge 正确数 / 1,333 | LLM-Judge (%) | F1 (%) | BLEU-1 (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-Max | Full Context Image | 1,333 | 624 / 1,333 | 46.81 | 8.35 | 4.58 |
| Qwen3.8-Max | ReMe Text | 1,330 | 511 / 1,333 | 38.33 | 4.53 | 2.39 |
| Qwen3.8-Max | ReMe Image | 1,332 | 809 / 1,333 | **60.69** | 6.52 | 3.48 |
| kimi-k3 | ReMe Text | 1,333 | 478 / 1,333 | 35.86 | 5.52 | 2.97 |
| kimi-k3 | ReMe Image（历史变体） | 1,333 | 682 / 1,333 | **51.16** | 7.43 | 4.02 |
| deepseek-v4.1-flash | ReMe Text | 1,333 | 525 / 1,333 | 39.38 | 5.43 | 2.96 |
| deepseek-v4.1-flash | ReMe Image | 1,333 | 749 / 1,333 | **56.19** | 7.14 | 3.87 |

ReMe Image 在三个模型上的 LLM-Judge 准确率均高于 ReMe Text。在 Qwen3.8-Max 上，Image 还比 Full Context 高 13.88 个百分点；Full Context 的 F1、BLEU-1 则更高。

Kimi、DeepSeek 的 Full Context 均为 **N/A**：完整历史超过所用服务每请求最多 256 张图片的限制，因此未继续评测。此前 OpenViking、Mem0 的 MobileMem 运行不在本次对照范围内。

## 方法与 Judge 设置 {#evaluation-settings}

| 方法 | 记忆输入与回答阶段的数据访问范围 |
| --- | --- |
| Full Context Image | 直接读取按时间排序的完整对话和历史图片。图片缩放为 256 × 256 的 RGB PNG，使用 LANCZOS 重采样（`paper256`）。 |
| ReMe Text | 使用原始对话文字构建记忆；通过 BM25 检索 daily、digest 笔记和原始 session。 |
| ReMe Image | 使用原始对话文字和图片构建记忆；检索范围与 Text 相同，并通过 `read_image` 重读附件。输入保留原分辨率；MPO 主图转换为 PNG。 |

历史文字、时间戳、角色和图片顺序均保留，Kimi Image 的两段会话排除另见下文。数据集另附的 caption、摘要、参考答案和证据标注不进入候选模型输入。回答保留官方 QA 提示词；ReMe Image 的记忆、回答 Job 均显式接收图片。

| 设置 | ReMe 记忆／压缩 | ReMe 回答 | Full Context reader | Judge |
| --- | --- | --- | --- | --- |
| 模型 | 表中候选模型 | 同一候选模型 | 表中候选模型 | Qwen3.8-Max |
| Thinking | 开启 | 开启 | 开启 | 关闭 |
| Temperature | 0 | 0 | 0 | 0 |
| 输出预算（tokens） | 65,536 | 65,536 | 65,536 | 4,096 |
| 上下文上限（tokens） | 1,000,000 | 1,000,000 | — | — |

ReMe 每题新建回答 Agent，最多 **30 轮**，采用原生上下文压缩，工具为 `search`、`read`、`read_image`。原始 session 仍可检索；图片适配的读取上限为 50 MiB，Agent 上下文最多保留 4,096 张图。

全部结果统一使用 **MemBase 默认 Judge** 提示词及原生解析器，模型和参数见上表。协议名为 `membase_default`，实现见 [MobileMem Judge](../../../../plugins/mobilemem-judge/README.md)。Qwen Full Context 的原预测使用该默认协议评分；更早的严格 rubric 分数不进入本表。参考答案和证据标注只提供给 Judge。

## Kimi Image 历史变体 {#kimi-history-variant}

Kimi Image 最终结果保留**此前成功的 1,203 份答案**，补齐**两个用户的其余 130 题**。这两个用户的记忆摄入分别排除 `uuid_8` 的 `8_69`（原始顺序从 0 计数的位置 127）和 `uuid_14` 的 `14_74_2`（位置 116）。全部 1,333 题仍纳入评测，其他 session 保持原顺序；表中将该行标为历史变体。

Kimi 恢复采用 session checkpoint、同 payload HTTP 重试，并提高了题目并发。除上述两段会话排除外，输入、提示词、reader／Judge 预算及 30 轮回答上限保持不变。DeepSeek 针对技术失败和内容审核错误进行了补跑。两个模型均保留第一份合格结果，包括有效 0 分。

## 结果来源

主表合并 Qwen 归档 `final-20261017` 与 Kimi、DeepSeek 追加归档 `append-20261023-kimi-flash`。后者的 `statistics/mobile-final.json` 保存最终指标和固定分母，`selected/mobile-selected.json` 保存逐题选择。本文聚合已有预测和评分，未新增模型调用。数据版本为 `989ba00900bddf4e0d4b0c7dd52f0840a7ee3a54`，[官方评测代码](https://github.com/zjunlp/MobileMem/tree/fa5cd13c9c659699df9caf650987ba2bea8ca085/omni/eval/eval)固定于 `fa5cd13c9c659699df9caf650987ba2bea8ca085`。
