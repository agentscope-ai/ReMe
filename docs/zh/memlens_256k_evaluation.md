# MemLens-256K 评测说明

本文介绍[多模态记忆文章](./blog_20261009.md#memlens-256k-results)中 MemLens-256K 评测的实验设置、数据筛选和结果。

## 结果与统计范围 {#results}

评测使用 [MemLens](https://huggingface.co/datasets/xiyuRenBill/MEMLENS) 官方 `agent_subset_195.json` 中的 195 个问题，每题使用对应的完整 256K 历史。按下文说明过滤六个问题后，所有方法在相同的 189 题上评测。LLM-Judge 按官方类别／子类别评分协议为每题给出 0 或 1 分，得分为 `100 × 正确数 / 189`。

| 模型 | 方法 | Judge 正确数 / 189 | LLM-Judge (%) |
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

ReMe Image 在两个模型上的得分分别为 **53.44%** 和 **72.49%**，比 Full Context Image 高 **6.35** 和 **6.88** 个百分点。

## 方法与 Judge 设置 {#evaluation-settings}

| 方法 | 记忆输入与回答阶段的数据访问范围 |
| --- | --- |
| Full Context Image | 回答模型直接读取完整历史文字与原图。 |
| ReMe Text | 使用原始会话文字生成记忆；回答 Agent 通过 BM25 检索 daily、digest 和原始 session，并读取相关文本。 |
| ReMe Image | 使用原始会话文字和图片生成记忆；检索范围与 Text 相同，并可通过 `read_image` 重新读取保存的原图。 |
| OpenViking | 使用原生图片描述和记忆流程；回答阶段检索限定为 memory，采用默认 top 10。 |
| Mem0 | 使用 Mem0 2.0.4 原生视觉入口生成记忆；回答阶段通过 FAISS 检索 top 20 记忆。 |

各方法分别使用 Qwen3.6-35B-A3B 和 Qwen3.8-Max 作为待测模型。ReMe Image 与 Mem0 均直接使用原图。ReMe 的记忆和回答上下文上限分别为 200,000 / 400,000 tokens，temperature 为 0，开启 thinking；记忆／压缩与回答输出预算分别为 65,536 / 16,384 tokens。每题使用独立回答 Agent，最多运行 30 轮，并采用原生上下文压缩。Image 的 `read_image` 支持读取保存的附件，读取上限为 50 MiB，图片保留上限为 4,096 张。

Full Context、OpenViking 和 Mem0 的回答阶段同样使用 temperature 0、thinking 开启、输出预算 16,384 tokens。Mem0 记忆阶段采用原生设置：输出预算 2,000 tokens，temperature 和 top_p 均为 0.1。OpenViking 的评测适配包含 JSON 解析修正，以及图片处理失败时的显式报错。

评分采用 [MemLens 官方协议](https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2)，所有方法统一使用 **Qwen3.8-Max** 作为 Judge，temperature 为 **0**、thinking **关闭**、输出预算为 **256 tokens**。Judge 与待测方法的记忆应用独立运行；参考答案及题型标签仅用于评分。

## 统一内容审核过滤 {#content-moderation-filter}

调用 Qwen 模型服务时，下表中的方法在处理相应问题时触发了内容安全审核（绿网协议）。我们对各方法、两个模型触发审核的问题取并集，共得到六题，并在所有方法和两个模型中统一过滤，最终统计相同的 **189/195** 题。

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
