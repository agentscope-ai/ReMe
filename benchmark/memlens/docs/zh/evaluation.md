# MemLens-256K 评测说明

本文介绍[多模态记忆文章](../../../../docs/zh/blog_20261009.md#memlens-256k-results)中 MemLens-256K 的实验设置、数据筛选和四模型结果。ReMe Image 的运行方式见 [benchmark README](../../README_ZH.md)。

## 结果与统计范围 {#results}

评测使用 [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS) 官方 `agent_subset_195.json` 中的 195 个问题及其对应的 256K 历史。其中六道题在部分方法的模型请求中触发了服务商的内容审核。为在相同题目上比较，四个模型、全部方法统一排除这六题，汇报其余 **189/195 题**的结果。

LLM-Judge 按官方类别／子类别协议为有效回答给出 0 或 1 分，准确率为 `100 × 正确数 / 189`。SubEM 判断规范化后的标准答案是否为预测文本的子串，F1 衡量规范化后 token 的重叠程度。各项指标先逐题计算，再对 189 题的得分取平均（macro average）。Kimi、DeepSeek 未完成的题目在汇总中按 0 计，表中单列有效结果数；已有答案和评分（含 0 分）保持原值。

| 模型 | 方法 | 有效结果数 / 189 | SubEM (%) | F1 (%) | Judge 正确数 / 189 | LLM-Judge (%) |
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

ReMe Image 在每个候选模型的五方法比较中均取得最高 LLM-Judge 准确率。

## 方法与 Judge 设置 {#evaluation-settings}

| 方法 | 记忆构建 | 回答流程 |
| --- | --- | --- |
| Full Context Image | 无。 | 模型直接读取完整历史文字与原图并回答。 |
| ReMe Text | 通过 ReMe 的 `auto_memory` 处理原始会话文字。 | 每题新建 `agentic_answer` Agent，通过 BM25 检索 daily、digest 和原始 session，多轮读取相关文本后回答。 |
| ReMe Image | 沿用相同记忆流程，同时接收原始文字和图片。 | 与 Text 使用相同的 Agent 流程，并可通过 `read_image` 重读保存的原图。 |
| OpenViking | 使用原生图片描述和 session 记忆流程。 | 调用一次原生 `search`，检索限定为 memory，沿用 SDK 默认 top 10；再由独立 reader 生成一次回答。 |
| Mem0 | 使用 Mem0 2.0.4 的原生视觉与记忆提取流程。 | 调用一次原生 `search`，通过 FAISS 取 top 20 记忆；再由独立 reader 生成一次回答。 |

基线保留各自的记忆构建与检索流程。OpenViking、Mem0 采用 retrieve → answer，ReMe 则使用多轮 `agentic_answer`。各行使用对应候选模型回答，需要构建记忆的方法也使用同一模型。ReMe Image 与 Mem0 接收原图，不使用 benchmark 附带的 caption；OpenViking 通过自己的图像描述流程处理图片。

评测服务器上的 OpenViking 主包版本为 **0.4.17.2.dev1**，从源码以 editable 方式安装，基于[该上游版本](https://github.com/volcengine/OpenViking/tree/420d4074f74070bc6fe7151aa54527cf1ff15152)，配套 Python SDK 为 0.1.9；检索正文按 MemLens 的普通 QA 模板组织。**Mem0 2.0.4** 同样从[锁定版本的源码](https://github.com/mem0ai/mem0/tree/75a37ec93db7278e3bd9aaf2aa3d6e5139e6789d)以 editable 方式安装，沿用 [MemLens 复现说明](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/README.md#mem0)中的流程和对应的 [QA 消息模板](https://github.com/xrenaf/MEMLENS/blob/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/memory-agent/prompt_builders.py)。

| 阶段 | 上下文上限 | 输出预算 | Thinking / temperature |
| --- | ---: | ---: | --- |
| ReMe 记忆／压缩 | 200,000 | 65,536 | 开启 / 0 |
| ReMe 回答 | 400,000 | 16,384 | 开启 / 0 |
| Full Context、OpenViking、Mem0 回答 | — | 16,384 | 开启 / 0 |
| MemLens Judge | — | 256 | 关闭 / 0 |

ReMe 每题使用独立回答 Agent，最多运行 30 轮，并采用上下文压缩。`read_image` 的附件读取上限为 50 MiB，图片保留上限为 4,096 张。

Mem0 记忆提取阶段的输出预算，**Qwen 为 2,000 tokens，Kimi、DeepSeek 为 10,000 tokens**。Qwen、DeepSeek 沿用 Mem0 2.0.4 的 [OpenAI 适配器默认配置](https://github.com/mem0ai/mem0/blob/75a37ec93db7278e3bd9aaf2aa3d6e5139e6789d/mem0/configs/llms/openai.py)：`temperature=0.1`、`top_p=0.1`，这两个值来自 Mem0 的配置。Kimi 所用服务不接受这两个参数，因此请求中不发送它们，服务端实际采用的默认值未知。记忆提取请求也不显式设置 thinking 开关，沿用模型服务的默认行为；回答阶段则按上表显式开启 thinking。

评分采用 [MemLens 官方协议](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2)，所有方法统一使用 **Qwen3.8-Max** 作为 Judge，temperature 为 **0**、thinking **关闭**、输出预算为 **256 tokens**。Judge 与待测方法的记忆应用独立运行；参考答案及题型标签仅用于评分。

## 统一内容审核过滤 {#content-moderation-filter}

下表列出 Qwen3.6-35B-A3B、Qwen3.8-Max 在记忆构建或回答阶段触发内容审核的六道题，以及对应的方法。四个模型、全部方法统一排除这六题，均在其余 189 题上统计。

表中 A3B 指 Qwen3.6-35B-A3B，Max 指 Qwen3.8-Max。

| 问题 ID | 触发内容审核的方法 / 模型 | 阶段 |
| --- | --- | --- |
| `q_028cb832` | OpenViking / Max | 记忆构建 |
| `q_1d99a36c` | OpenViking / A3B | 记忆构建 |
| `q_3b01c02a` | ReMe Text、Image / A3B；OpenViking / A3B、Max；Mem0 / A3B | 记忆构建 |
| `q_564a657b` | ReMe Image / Max | 回答 |
| `q_600a79aa` | ReMe Image / Max | 记忆构建 |
| `q_c9f8e24d` | ReMe Text / Max | 回答 |

## 独立复现结果 {#independent-reproduction}

我们将 MemLens benchmark 代码整理并集成到 ReMe 项目中，在相同的数据输入和评分设置下，对 ReMe Image 进行了独立复现。运行方式见[评测说明](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/benchmark/memlens/README.md)，具体参数见[方法配置](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens/src/reme_memlens/benchmark.yaml)、[Judge 配置](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens-judge/src/judge_memlens/plugin.yaml)和[Judge 评分规则](https://github.com/agentscope-ai/ReMe/blob/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8/plugins/memlens-judge/src/judge_memlens/protocol.py)。

独立复现对应的公开评测代码见 [PR #640](https://github.com/agentscope-ai/ReMe/pull/640)，commit 为 [962d5cd](https://github.com/agentscope-ai/ReMe/commit/962d5cd8401c1d2be17f3bb95f8ac9c185672cd8)。

应用相同的六题过滤后，复现实验中 A3B 有 1 题、Max 有 2 题因会话历史处理失败未得到完整结果。下表分别在两次实验均完成的 188 题和 187 题上进行配对比较。

| 模型 | 配对题数 | 原实验正确数 | 独立复现正确数 | 原实验 → 独立复现 LLM-Judge (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 188 | 100 | 96 | 53.19 → 51.06 |
| Qwen3.8-Max | 187 | 135 | 134 | 72.19 → 71.66 |

在上述配对题目上，Qwen3.6-35B-A3B 和 Qwen3.8-Max 的独立复现 LLM-Judge 得分分别比原实验低 2.13 和 0.53 个百分点，整体性能与原实验基本保持一致。
